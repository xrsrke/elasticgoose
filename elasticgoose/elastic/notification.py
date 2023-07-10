from typing import Protocol


class NotificationReceiver(Protocol):
    def on_hosts_updated(self): pass


class NotificationManager:
    def __init__(self):
        pass


class NotificationService:
    pass


class NotificationClient:
    pass
