from aiortc import VideoStreamTrack


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()

        img = frame.to_ndarray(format="bgr24")

        # тут можно преобразовать изображение, если нужно

        return img
