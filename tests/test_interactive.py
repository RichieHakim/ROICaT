from pathlib import Path

import warnings
import pytest
import tempfile

import numpy as np
import torch
import multiprocessing as mp

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


def create_mock_input():
    ## Create mock input. Looks dumb. But it's just a test.
    mock_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    mock_idx_images_overlay = torch.tensor([0, 1, 2, 3])
    mock_images_overlay = np.random.rand(4, 2, 2)
    return mock_data, mock_idx_images_overlay, mock_images_overlay


def get_indices():
    ## Steal the fn_get_indices function for testing purposes
    path_tempFile = tempfile.gettempdir() + "/indices.csv"
    with open(path_tempFile, "r") as f:
        indices = f.read().split(",")
    indices = [int(i) for i in indices if i != ""] if len(indices) > 0 else None
    return indices


def start_server(apps):
    ## Start Bokeh server given a test scatter plot
    server = Server(
        apps,
        port=5000,
        address="0.0.0.0",
        allow_websocket_origin=["0.0.0.0:5000", "localhost:5000"],
    )
    server.start()

    ## Setup IO loop
    server.io_loop.start()


def deploy_bokeh(instance):
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


def test_interactive_drawing():
    warnings.warn("Interactive GUI Drawing Test is running. Please wait...")
    ## Bokeh server deployment at http://localhost:5000
    apps = {"/": Application(FunctionHandler(deploy_bokeh))}

    warnings.warn("Deploy Bokeh server to localhost:5000...")
    ## Let it run in the background so that the test can continue
    server_process = mp.Process(target=start_server, args=(apps,))
    server_process.start()

    warnings.warn("Setup chrome webdriver...")
    service = Service()
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1280,1280")
    ## For local testing, just comment out the headless options.
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    ## if you are on latest version say selenium v4.6.0 or higher, you don't have to use third party library such as WebDriverManager
    driver = webdriver.Chrome(service=service, options=chrome_options)

    warnings.warn("Get to the Bokeh server...")
    driver.get("http://localhost:5000/")
    wait = WebDriverWait(driver, 10)
    try:
        element = wait.until(EC.presence_of_element_located((By.XPATH, "//*")))
        warnings.warn("Found Bokeh drawing element!")
    except Exception as e:
        warnings.warn(f"Failed to locate element: {str(e)}")

    ## Create movement set
    size = element.size
    width, height = size["width"], size["height"]

    ## Move to the center of the element
    warnings.warn("Start mouse movement...")
    actions = ActionChains(driver)
    actions.move_to_element(element)
    actions.click_and_hold()

    ## Draw!
    actions.move_by_offset(
        int(width / 2), int(0)
    )  ## Move from center to midpoint of right edge
    actions.move_by_offset(
        int(0), int(-height / 2)
    )  ## Move from midpoint of right edge to top right corner
    actions.move_by_offset(
        int(-width / 2), int(0)
    )  ## Move from top right corner to midpoint of top edge
    actions.release()
    actions.perform()

    warnings.warn("Mouse movement done! Detach Selenium from Bokeh server...")
    driver.quit()

    warnings.warn("Test if indices are correctly saved...")
    indices = get_indices()
    assert indices == [3]

    warnings.warn("Test is done. Cleaning up...")
    server_process.terminate()
    server_process.join()
    warnings.warn("Test is done. Cleaning up done.")
