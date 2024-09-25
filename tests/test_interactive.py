import os
import sys
import pytest
import tempfile
import shutil
import requests
import socket
import psutil
import time
from functools import partial

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Queue

from roicat import visualization
import holoviews as hv
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

hv.extension("bokeh")

def get_tempdir_free_space():
    temp_dir = tempfile.gettempdir()
    usage = shutil.disk_usage(temp_dir)
    return usage.free


def create_mock_input():
    ## Create mock input. Looks dumb. But it's just a test.
    mock_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    mock_idx_images_overlay = torch.tensor([0, 1, 2, 3])
    mock_images_overlay = np.random.rand(4, 2, 2)
    return mock_data, mock_idx_images_overlay, mock_images_overlay


def get_indices(path_tempFile):
    ## Steal the fn_get_indices function from roicat.visualization for testing purposes
    with open(path_tempFile, "r") as f:
        indices = f.read().split(",")
    indices = [int(i) for i in indices if i != ""] if len(indices) > 0 else None
    return indices


def deploy_bokeh(instance, tmp_path):
    ## Draw test plot and add to Bokeh document
    hv.extension("bokeh")

    ## Create a mock input
    mock_data, mock_idx_images_overlay, mock_images_overlay = create_mock_input()

    ## Create a scatter plot
    _, layout, _ = visualization.select_region_scatterPlot(
        data=mock_data,
        idx_images_overlay=mock_idx_images_overlay,
        images_overlay=mock_images_overlay,
        size_images_overlay=0.01,
        frac_overlap_allowed=0.5,
        path=tmp_path,
        figsize=(1200, 1200),
        alpha_points=1.0,
        size_points=10,
        color_points="b",
    )

    ## Render plot
    hv_layout = hv.render(layout)
    hv_layout.name = "drawing_test"

    ## Add to Bokeh document
    instance.add_root(hv_layout)


def is_port_available(port, host='localhost'):
    """Check if a given port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind((host, port))
        except socket.error:
            return False
        return True


def start_server(apps, query):
    ## Start Bokeh server given a test scatter plot
    server = Server(
        apps,
        port=5006,
        address="0.0.0.0",
        allow_websocket_origin=["0.0.0.0:5006", "localhost:5006"],
    )
    server.start()

    ## Put server details into the queue
    query.put(
        {
            "address": server.address,
            "port": server.port,
        }
    )

    ## Setup IO loop
    server.io_loop.start()


def kill_process_on_port(port):
    for proc in psutil.process_iter(attrs=["pid", "name"]): ## Updated to new psutil API: 2024.09.24
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    try:
                        psutil.Process(proc.info["pid"]).terminate()
                        return True
                    except psutil.AccessDenied:
                        return False
        except (psutil.AccessDenied, psutil.ZombieProcess):
            ## Happens when you don't have permission to access the process
            pass
    return False


def check_server():
    try:
        response = requests.get("http://localhost:5006/test_drawing")
        if response.status_code == 200:
            print("Server is up and running!")
            return 1
        else:
            print(f"Server responded with status code: {response.status_code}")
            return 0
    except requests.ConnectionError:
        print("Cannot connect to the server!")
        return 0


def test_interactive_drawing():
    print("Interactive GUI Drawing Test is running. Please wait...")
    ## Parameter setting
    port = 5006
    server_iter = 0
    max_server_iter = 3
    max_webdriver_iter = 3
    path_tempdir = tempfile.gettempdir()
    path_tempfile = os.path.join(path_tempdir, "indices.csv")

    ## Make sure the tempdir has enough space
    print("Check if tempdir has enough space...")
    free_space = get_tempdir_free_space()
    print(f"Free space in tempdir: {free_space / (1024**3):.2f} GB")

    ## Start iteration. Iteration will stop if indices.csv is created.
    while server_iter <= max_server_iter:
        ## First, check if the port for Bokeh server is available
        if is_port_available(port):
            print(f"Port {port} is available!")
        else:
            print(f"Port {port} is not available!")
            print(f"Kill process using port {port}...")
            kill_process_on_port(port)

        ## Bokeh server deployment at http://localhost:5006/test_drawing
        partial_deploy_bokeh = partial(deploy_bokeh, tmp_path=path_tempfile)
        apps = {"/test_drawing": Application(FunctionHandler(partial_deploy_bokeh))}

        print("Deploy Bokeh server to localhost:5006/test_drawing...")
        ## Let it run in the background so that the test can continue
        query = Queue()
        server_process = mp.Process(target=start_server, args=(apps, query))
        server_process.start()

        ## Wait for the server to start
        time.sleep(5)

        ## Get server info
        server_query = query.get()
        print(f"Server address: {server_query['address']}")
        print(f"Server port: {server_query['port']}")

        ## Prevent permanant hang: If bug occurs, kill the server
        try:
            ## Check if the server is up and running
            print("Check if Bokeh server is up and running...")
            server_status = check_server()
            if not server_status:
                server_process.terminate()
                server_process.join()
                raise Exception("Server is not up and running!")

            print("Setup chrome webdriver...")
            service = Service()
            chrome_options = Options()
            chrome_options.add_argument("--window-size=1280,1280")

            ## For local testing, comment out these options to visualize actions in bokeh / selenium server.
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")

            ## if you are on latest version say selenium v4.6.0 or higher, you don't have to use third party library such as WebDriverManager
            print("Driver parameter sanity check...")
            driver = webdriver.Chrome(service=service, options=chrome_options)
            capabilities = driver.capabilities
            print("Browser Name: {}".format(capabilities.get("browserName")))
            print("Browser Version: {}".format(capabilities.get("browserVersion")))
            print("Platform Name: {}".format(capabilities.get("platformName")))
            print(
                "Chrome Driver Version: {}".format(
                    capabilities.get("chrome").get("chromedriverVersion")
                )
            )

            print("Get to the Bokeh server...")
            driver.get("http://localhost:5006/test_drawing")
            wait = WebDriverWait(driver, 10)
            print("Found the Bokeh server, locate drawing Bokeh element...")
            try:
                element = wait.until(EC.presence_of_element_located((By.XPATH, "//*")))
                print("Found Bokeh drawing element!")
            except Exception as e:
                print(f"Failed to locate element: {str(e)}")

            ## Create movement set
            size = element.size
            width, height = size["width"], size["height"]

            print("element size: {}".format(size))
            print("element tagname: {}".format(element.tag_name))
            print("element text: {}".format(element.text))
            print("element location: {}".format(element.location))
            print("element displayed: {}".format(element.is_displayed()))
            print("element enabled: {}".format(element.is_enabled()))
            print("element selected: {}".format(element.is_selected()))

            ## iterate over actions until indices.csv is created
            webdriver_iter = 0
            while webdriver_iter <= max_webdriver_iter:
                print(f"Action iteration {webdriver_iter} starts...")
                ## Move to the center of the element
                print("Start mouse movement...")
                actions = ActionChains(driver)

                ## Surprisingly, this pause seems to be crucial.
                actions.pause(30 * (webdriver_iter + 1))

                ## Move to the center of the element
                actions.move_to_element(element)
                actions.click_and_hold()

                ## Draw!
                actions.move_by_offset(
                    int(width / 2), int(0)
                )  ## Move from center to midpoint of right edge
                actions.move_by_offset(
                    int(0), int(-height / 2)*0.9
                )  ## Move from midpoint of right edge to top right corner
                actions.move_by_offset(
                    int(-width / 2), int(0)
                )  ## Move from top right corner to midpoint of top edge
                actions.release()
                actions.perform()

                print("Mouse movement done!")

                ## Wait for the server to save indices.csv
                time.sleep(5)

                ## Test if indices.csv is created before we kill the server
                if os.path.exists(path_tempfile):
                    print("indices.csv is created! Detach Selenium from Bokeh server...")
                    driver.quit()
                    break
                else:
                    print("indices.csv is not created! Repeat the interaction...")
                    webdriver_iter += 1
        except Exception as e:
            print(f"Exception occured: {e}, Kill the Bokeh server...")
            server_process.terminate()
            server_process.join()

        ## Kill the process to prevent potential race condition
        print("Kill the Bokeh server...")
        server_process.terminate()
        server_process.join()

        ## Wait for the process to terminate
        time.sleep(5)

        ## If indices.csv is created, break the loop
        if os.path.exists(path_tempfile):
            print("indices.csv is created!")
            break
        else:
            print("indices.csv is not created!")
            server_iter += 1

    print("Test if indices.csv has correct permission...")
    if not os.access(path_tempfile, os.R_OK):
        print("indices.csv is not readable!")
        server_process.terminate()
        server_process.join()
        raise Exception("indices.csv is not readable!")

    print("Test if indices are correctly saved...")
    indices = get_indices(path_tempfile)

    if indices is None:
        print("indices.csv is created, but no indices are saved!")
        server_process.terminate()
        server_process.join()
        raise Exception("indices.csv is created, but no indices are saved!")

    ## Check if the indices are correct
    assert indices == [3]
    print("Test is done.")
