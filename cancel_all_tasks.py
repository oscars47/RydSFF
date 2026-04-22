## run if you accidentally submitted tasks and need to cancel all of them
## will cancel all tasks in the QUEUED state in AWS Braket.

import boto3
from botocore.exceptions import ClientError

def cancel_all_queued_tasks(region_name=None):
    """
    Cancel every AWS Braket quantum task in the QUEUED state
    by paging through search_quantum_tasks.
    """
    client = boto3.client("braket", region_name=region_name)

    # Set up the paginator for search_quantum_tasks
    paginator = client.get_paginator("search_quantum_tasks")

    # Define the filter to only grab QUEUED tasks
    status_filter = [{
        "name": "status",         # filter by task status
        "operator": "EQUAL",      # exact match
        "values": ["QUEUED"]      # only QUEUED
    }]

    cancelled_count = 0

    # Paginate through all QUEUED tasks (up to 100 per page)
    for page in paginator.paginate(
        filters=status_filter,
        PaginationConfig={"PageSize": 100}
    ):
        for task_summary in page.get("quantumTasks", []):
            arn = task_summary["quantumTaskArn"]
            try:
                client.cancel_quantum_task(quantumTaskArn=arn)
                cancelled_count += 1
                print(f"Cancelled: {arn}")
            except ClientError as e:
                # handle throttling or other errors
                print(f"Error cancelling {arn}: {e.response['Error']['Message']}")

    print(f"\nTotal tasks cancelled: {cancelled_count}")
    return cancelled_count

if __name__ == "__main__":
    # Optionally override default region:
    # cancel_all_queued_tasks(region_name="eu-west-1")
    cancel_all_queued_tasks()
