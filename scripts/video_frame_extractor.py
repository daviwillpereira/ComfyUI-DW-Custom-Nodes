import cv2
from pathlib import Path

class VideoFrameExtractor:
    """
    Handles the extraction of individual frames from a video file.
    Designed to process short clips for manual frame-by-frame editing pipelines.
    """
    
    def __init__(self, video_path: str, output_directory: str):
        self.video_path = Path(video_path)
        self.output_directory = Path(output_directory)

    def extract_frames(self) -> None:
        """
        Reads the video file and saves each frame sequentially as a high-quality PNG.
        Raises FileNotFoundError if the source video does not exist.
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Source video not found at: {self.video_path}")

        # Ensure the output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

        video_capture = cv2.VideoCapture(str(self.video_path))
        frame_counter = 0

        while True:
            success, frame = video_capture.read()
            
            if not success:
                break

            # Pad the filename with zeros (e.g., frame_0001.png) to maintain correct sorting
            frame_filename = self.output_directory / f"frame_{frame_counter:04d}.png"
            
            # Save as PNG to avoid compression artifacts during manual editing
            cv2.imwrite(str(frame_filename), frame)
            frame_counter += 1

        video_capture.release()
        print(f"Extraction complete. {frame_counter} frames successfully exported to '{self.output_directory}'.")


if __name__ == "__main__":
    # Execution setup for the 0.5s video extraction
    extractor = VideoFrameExtractor(
        video_path="download.mp4", 
        output_directory="extracted_frames"
    )
    extractor.extract_frames()