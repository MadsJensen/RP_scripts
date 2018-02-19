import os
from slackclient import SlackClient

sc = SlackClient(os.environ.get('SLACK_TOKEN'))
channel = "#decoding_intentions"


def send_slack(string_to_send, channel=channel):
    sc.api_call(
        "chat.postMessage",
        channel=channel,
        text=string_to_send,
    )
