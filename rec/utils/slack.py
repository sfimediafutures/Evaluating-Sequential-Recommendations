import os
import requests
import traceback
import time


class Slack():
    def __init__(self) -> None:
        self.slack_url = os.getenv('SLACK_URL')
        if not self.slack_url:
            raise ValueError('SLACK_URL environment variable not found.')
        
        self.slack_channel = os.getenv('SLACK_CHANNEL')
        if not self.slack_channel:
            raise ValueError('SLACK_CHANNEL environment variable not found.')
        
    def send_message(self, message):
        payload = {
            "channel": self.slack_channel,
            "text": message
        }
        requests.post(self.slack_url, json=payload)

    def send_exception(self, exc_info):
        exc_type, exc_value, exc_traceback = exc_info
        payload = {
            "channel": self.slack_channel,
            "text": exc_type.__name__,
            "attachments": [
                {
                    "color": "#2eb886",
                    "text": "`" + str(exc_value) + "`"
                }
            ],
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": exc_type.__name__,
                    }
                },
                {
                    "type": 'section',
                    "text": {
                        "type": 'mrkdwn',
                        "text": '```' + ''.join(traceback.format_tb(exc_traceback)) + '```'
                    }
                },
            ]
        }
        requests.post(self.slack_url, json=payload)

    def send_results(self, message, avg_ctr, mean_avg_precision, avg_popularity_score, avg_count_popularity_score, coverage, gini_index):
        payload = {
            'text': message,
            'attachments': [
                {
                    'fallback': 'Attachment fallback text',
                    'color': '#36a64f',
                    'pretext': 'Heres results for the evaluation:',
                    'fields': [ 
                        {'title': 'Average CTR', 'value': avg_ctr, 'short': True},
                        {'title': 'Mean Average Precision', 'value': mean_avg_precision, 'short': True},
                        {'title': 'Average Duration Popularity Score', 'value': avg_popularity_score, 'short': True},
                        {'title': 'Average Count Popularity Score', 'value': avg_count_popularity_score, 'short': True},
                        {'title': 'Coverage', 'value': coverage, 'short': True},
                        {'title': 'Gini Index', 'value': gini_index, 'short': True},
                    ],
                    'footer': 'Attachment Footer',
                    'ts': int(time.time())
                }
            ]
        }

        requests.post(self.slack_url, json=payload)


    # def upload_file(self, file_path, message=""):
    #     # Get the file size and filename
    #     file_size = os.path.getsize(file_path)
    #     filename = os.path.basename(file_path)

    #     # Step 1: Request an upload URL from Slack
    #     url = "https://slack.com/api/files.getUploadURLExternal"
    #     headers = {
    #         "Authorization": f"Bearer {self.slack_token}",
    #         "Content-Type": "application/x-www-form-urlencoded"
    #     }
    #     data = {
    #         "filename": str(filename),
    #         "length": file_size
    #     }

    #     try:
    #         response = requests.post(url, headers=headers, data=data)
    #         if not response.ok:
    #             raise ValueError(f"Error requesting upload URL: {response.text}")

    #         response_data = response.json()
    #         if not response_data.get("ok"):
    #             raise ValueError(f"Error in response data: {response_data}")

    #         upload_url = response_data["upload_url"]
    #         file_id = response_data["file_id"]

    #         # Step 2: Use the received URL to upload the file
    #         with open(file_path, "rb") as file_content:
    #             files = {'file': (filename, file_content)}
    #             upload_response = requests.post(upload_url, files=files)
    #             if not upload_response.ok:
    #                 raise ValueError(f"Error uploading file: {upload_response.text}")

    #         # Step 3: Notify Slack about the file upload completion
    #         complete_url = "https://slack.com/api/files.completeUploadExternal"
    #         complete_data = {
    #             "files": {"id":file_id,"title":filename},
    #             "channel_id": self.slack_channel,
    #         }
    #         complete_response = requests.post(complete_url, headers=headers, data=complete_data)
    #         print(complete_response.json())
    #         if not complete_response.ok:
    #             raise ValueError(f"Error completing file upload: {complete_response.text}")

    #         complete_response_data = complete_response.json()
    #         if not complete_response_data.get("ok"):
    #             raise ValueError(f"Error in complete response data: {complete_response_data}")

    #         print(f"File uploaded successfully: {file_id}")
    #     except Exception as e:
    #         print(f"Error uploading file: {e}")
