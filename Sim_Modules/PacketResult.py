from enum import Enum

class PacketResult(Enum):
    PACKET_OK = 1
    QUEUE_FULL = 2
    CHANNEL_ERROR = 3
