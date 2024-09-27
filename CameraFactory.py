#from PiVideoStream import PiVideoStream

class CameraFactory(object):

    """Class to select camera or video source"""

    def __init__(self, video_factory=None):
        """camera_factory is our abstract factory.  We can set it at will."""

        self.camera_factory = video_factory

    def get_image(self):
        """Creates and shows a pet using the abstract factory"""

        img = self.camera_factory.read()
        return img
    
    def stop(self):
        self.camera_factory.stop()