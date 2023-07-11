from elasticgoose.elastic.notification import NotificationManager

import pytest


@pytest.mark.skip
def test_notification_manager():
    class NotificationReceiver:
        def __init__(self):
            self.events = []

        def receive(self, timestamp, res):
            self.events.append((timestamp, res))

    manager = NotificationManager()
    manager.init()

    notification_receiver = NotificationReceiver()
    manager.register_listener(notification_receiver)
