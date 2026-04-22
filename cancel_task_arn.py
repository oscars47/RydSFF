import sys
from braket.aws import AwsQuantumJob

def main():
    if len(sys.argv) != 2:
        print("Usage: python cancel.py <JOB_ARN>")
        sys.exit(1)

    job_arn = sys.argv[1]

    job = AwsQuantumJob(arn=job_arn)
    job.cancel()

    print(f"Cancellation requested for job:\n{job_arn}")

if __name__ == "__main__":
    main()

    # use case: python cancel.py arn:aws:braket:us-east-1:442827106149:job/cf7c9461-9f9d-4355-9082-a4456e706ce7
