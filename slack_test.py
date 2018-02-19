import os
from slackclient import SlackClient

sc = SlackClient(os.environ.get('SLACK_TOKEN'))


def send_slack(string_to_send):
    sc.api_call(
        "chat.postMessage",
        channel="#dbs_notifications",
        text=string_to_send,
    )
