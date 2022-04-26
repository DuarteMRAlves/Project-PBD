import argparse
import grpc
import open_pose_pb2 as pb
import open_pose_pb2_grpc as pb_grpc
import typing


class KeypointInfo(typing.TypedDict):
    x: float
    y: float
    score: float


KeypointId = int

Keypoint = typing.Dict[KeypointId, KeypointInfo]

DetectedPose = typing.List[Keypoint]

DetectedPoses = typing.List[DetectedPose]


def parse_args():
    """Parse arguments for test setup.

    Returns: The arguments for the test
    """
    parser = argparse.ArgumentParser(description='Test for OpenPose gRPC Service')
    parser.add_argument(
        'image',
        help='Path to the image to send to the server')
    parser.add_argument(
        '--address',
        metavar='address',
        default='companhia.isr.tecnico.ulisboa.pt:4895',
        help='Location of the open pose server '
             '(defaults to companhia.isr.tecnico.ulisboa.pt:4895)')
    return parser.parse_args()


class OpenPoseClient:
    """Detects poses by calling an OpenPose grpc server.

    Args:
        address: address where the server is running.
    """

    def __init__(self, address: str):
        self.__address = address
        self.__channel = grpc.insecure_channel(self.__address)
        self.__stub = pb_grpc.OpenPoseEstimatorStub(self.__channel)

    def get_keypoints(self, *images: bytes) -> typing.List[DetectedPoses]:
        """Detects poses calling an OpenPose grpc server.

        Args:
            images: List of images as bytes.

        Returns: Detected poses for each image. Returns a List where each element is
        another list with the detected poses for the respective image. Each detected
        pose is a dictionary with the keypoint id as key. The value for each key is a
        dictionary with the keys "x", "y" and "score", for the relative x coordinate
        (between 0 and 1), relative y coordinate (between 0 and 1) and a confidence
        score for the system.
        """
        requests = [pb.Image(data=image) for image in images]
        replies = [self.__stub.estimate(request) for request in requests]
        return [self.reply_bp_to_dicts(reply) for reply in replies]

    @staticmethod
    def reply_bp_to_dicts(reply: pb.DetectedPoses) -> DetectedPoses:
        poses: DetectedPoses = []
        for pb_pose in reply.poses:
            pose: DetectedPose = {}
            for pb_kp in pb_pose.key_points:
                pose[pb_kp.index] = {
                    "x": pb_kp.x,
                    "y": pb_kp.y,
                    "score": pb_kp.score,
                }
            poses.append(pose)
        return poses

    def close(self):
        self.__channel.close()


def read_image(image_path: str) -> bytes:
    with open(image_path, "rb") as fp:
        return fp.read()


def main():
    args = parse_args()
    client = OpenPoseClient(address=args.address)
    try:
        # Read image from disk
        image = read_image(args.image)
        # Send an image and receive the detected poses.
        poses = client.get_keypoints(image)
        print("The detected poses where ", poses)
    # Print the errors if they occur
    except grpc.RpcError as rpc_error:
        print('An error has occurred:')
        print(f'  Error Code: {rpc_error.code()}')
        print(f'  Details: {rpc_error.details()}')


if __name__ == "__main__":
    main()
