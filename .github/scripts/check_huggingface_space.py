from typing import Optional

import argparse
import time

from huggingface_hub import HfApi

## Parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Check the status of a Huggingface space.")
    parser.add_argument(
        "--name",
        type=str,
        help="The name of the Huggingface space to check. Should be in the format 'owner/space_name'.",
        required=True,
    )
    ## --token is a string, None if not provided
    parser.add_argument(
        "--token",
        type=str,
        help="The Huggingface API token.",
        default=None,
    )
    parser.add_argument(
        "--max_wait_time",
        type=int,
        help="The maximum time to wait for the runtime to start.",
        default=300,
    )
    ## --restart_space is a flag argument, False by default
    parser.add_argument(
        "--restart_space",
        action="store_true",
        help="Whether to restart the space if it is not running.",
    )
    ## --error_on_restart is a flag argument, False by default
    parser.add_argument(
        "--error_on_failure",
        action="store_true",
        help="Whether to throw an error if the space is restarted.",
    )
    return parser.parse_args()


def check(
    huggingface_space_name: str,
    token: Optional[str] = None,
    max_wait_time: int = 300,
    restart_space: bool = True,
    error_on_failure: bool = False,
):

    hf_api = HfApi(
        token=token,
    )

    expected_runtime_status = ["RUNNING",]

    get_runtime_status = lambda: hf_api.get_space_runtime(repo_id=huggingface_space_name).stage

    ## If runtime not running, restart it
    runtime_status_initial = get_runtime_status()
    if (runtime_status_initial.upper() not in expected_runtime_status):
        if restart_space:
            hf_api.restart_space(repo_id=huggingface_space_name)

            ## Wait for the runtime to start
            tic = time.time()
            while (get_runtime_status() not in expected_runtime_status) or (time.time() - tic < max_wait_time):
                time.sleep(5)

        ## Throw an error to trigger a notification
        if error_on_failure:
            raise Exception(f"Found initial runtime status: {runtime_status_initial}. Expected: {expected_runtime_status}. Current status: {get_runtime_status()}.")
    else:
        print(f"Runtime status is {runtime_status_initial}. No action required.")


def main():
    args = parse_args()
    check(
        huggingface_space_name=args.name,
        token=args.token,
        max_wait_time=args.max_wait_time,
        restart_space=args.restart_space,
        error_on_failure=args.error_on_failure,
    )


if __name__ == "__main__":
    main()