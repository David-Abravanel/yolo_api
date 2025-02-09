from collections import defaultdict
import cv2
import shutil
import numpy as np
from typing import List, Tuple

from modules import Detection, Shape, Xyxy


class MaskService:

    @staticmethod
    def create_combined_mask(image_shape: tuple, masks: List[Shape], is_focus: bool = False) -> np.ndarray:
        """
        Creates a combined mask by merging multiple masks into one binary mask.

        Args:
            image_shape: Shape of the image (height, width).
            masks: List of masks to combine.
            is_focus: If True, keeps focus areas; if False, keeps exclusion areas.

        Returns:
            Combined binary mask (numpy array).
        """
        if masks:
            mask_regions = np.zeros(image_shape[:2], dtype=np.uint8)

            for mask_config in masks:
                temp_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                points = [
                    [int(coord.x * image_shape[1]),
                     int(coord.y * image_shape[0])]
                    for coord in mask_config.shape
                ]
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(temp_mask, [points], 1)
                mask_regions = cv2.bitwise_or(mask_regions, temp_mask)

            combined_mask = mask_regions if is_focus else 1 - mask_regions
            return combined_mask

        # return np.ones(image_shape[:2], dtype=np.uint8)
        return []

    @staticmethod
    def get_detections_on_mask(
        detections: List[Detection],
        mask: np.ndarray,
        shape: List[int],
        min_x: int = 10,
        min_y: int = 10
    ) -> List[Detection]:
        """
        Filters detections based on whether their center is inside the provided mask.

        Args:
            detections: List of detection objects (bounding boxes).
            mask: Binary mask to check against.
            shape: Shape of the image (height, width).
            min_x: min width of the bbox detection area to consider, default = 15
            min_y: min hight of the bbox detection area to consider, default = 15

        Returns:
            List of detections whose center is inside the mask.
        """
        if isinstance(mask, np.ndarray):
            ret_detections = []
            img_y, img_x = shape[:2]

            for detect in detections:
                x1, x2, y1, y2 = detect.bbox.x1, detect.bbox.x2, detect.bbox.y1, detect.bbox.y2

                x1, x2 = max(0, min(x1, img_x - 1)), max(0, min(x2, img_x - 1))
                y1, y2 = max(0, min(y1, img_y - 1)), max(0, min(y2, img_y - 1))

                points = [
                    # (int(x * img_x), int(y * img_y))
                    (int(x), int(y))
                    for i, x in enumerate([x1, (x1 + x2) / 2, x2])
                    for j, y in enumerate([y1, (y1 + y2) / 2, y2])
                    if not (i + j) % 2 == 0
                ]

                # if any(0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[int(y), int(x)] > 0 for x, y in points) and (x2-x1 > min_x) and (y2-y1 > min_y):
                if (x2-x1 > min_x) and (y2-y1 > min_y):
                    if isinstance(mask, np.ndarray) and not any((mask[int(y), int(x)] > 0) for x, y in points):
                        continue

                    detect.bbox = detect.bbox.update(
                        x1=x1/img_x, y1=y1/img_y, x2=x2/img_x, y2=y2/img_y)
                    ret_detections.append(detect)

            return ret_detections

        return detections

    def _validate_inputs(frames: List[np.ndarray], mask: np.ndarray = None) -> None:
        """
        Validate input frames and mask.

        Args:
            frames: List of input frames
            mask: Optional binary mask

        Raises:
            ValueError: If inputs are invalid
        """
        if len(frames) < 2:
            raise ValueError("At least two frames are required")

        frame_shape = frames[0].shape
        if not all(frame.shape == frame_shape for frame in frames):
            raise ValueError("All frames must have the same shape")

        if isinstance(mask, np.ndarray) and mask.shape[:2] != frame_shape[:2]:
            raise ValueError("Mask shape must match frame shape")

    def _is_valid_contour(contour: np.ndarray, mask: np.ndarray, min_area: int) -> bool:
        """
        Check if contour is valid based on area and mask.

        Args:
            contour: Contour to check
            mask: Binary mask for validation
            min_area: Minimum area threshold

        Returns:
            Boolean indicating if contour is valid
        """
        if cv2.contourArea(contour) < min_area:
            return False

        if isinstance(mask, np.ndarray):
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return False
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            if not (0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] == 1):
                return False

        return True

    def _create_bbox_masks(contours: List[np.ndarray], frame_shape: Tuple[int, ...],
                           box_padding: int, ret_color: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binary and color masks from contours using bounding boxes.

        Args:
            contours: List of valid contours
            frame_shape: Shape of the frame
            box_padding: Padding for bounding boxes

        Returns:
            Tuple of (binary_mask, color_mask)
        """
        height, width = frame_shape[:2]
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        color_mask = np.zeros((height, width, 3),
                              dtype=np.uint8) if ret_color else None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x1 = max(0, x - box_padding)
            y1 = max(0, y - box_padding)
            x2 = min(width, x + w + box_padding)
            y2 = min(height, y + h + box_padding)

            cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 1, -1)
            if ret_color:
                cv2.rectangle(color_mask, (x1, y1), (x2, y2), (0, 255, 0), -1)

        return binary_mask, color_mask if ret_color else binary_mask

    def _create_motion_accumulator(
        frames: List[np.ndarray],
        sensitivity: float,
        noise_threshold: int = 15,
        blur_strength: float = 1.0,
        temporal_smoothing: float = 0.5
    ) -> np.ndarray:
        """
        יצירת מסכת תנועה משופרת עם פרמטרים נוספים לשליטה.

        Args:
            frames: רשימת פריימים
            sensitivity: רגישות זיהוי תנועה (0.0 עד 1.0)
            noise_threshold: סף לסינון רעש (ערכים נמוכים יותר = רגישות גבוהה יותר)
            blur_strength: עוצמת הטשטוש (1.0 = רגיל, גבוה יותר = טשטוש חזק יותר)
            temporal_smoothing: החלקה זמנית בין פריימים (0.0 עד 1.0)

        Returns:
            מסכת תנועה מצטברת
        """
        # חישוב ערכי סף דינמיים
        base_threshold = noise_threshold + int(sensitivity * 50)
        blur_kernel_size = max(int(15 * blur_strength), 3)
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

        motion_accumulator = np.zeros(frames[0].shape[:2], dtype=np.float32)
        prev_motion = np.zeros_like(motion_accumulator)

        for i in range(len(frames) - 1):
            # המרה לגווני אפור לפני חישוב ההפרש
            frame1_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            # טשטוש לפני חישוב ההפרש לסינון רעש
            frame1_blur = cv2.GaussianBlur(
                frame1_gray, (blur_kernel_size, blur_kernel_size), 0)
            frame2_blur = cv2.GaussianBlur(
                frame2_gray, (blur_kernel_size, blur_kernel_size), 0)

            # חישוב הפרש
            diff = cv2.absdiff(frame1_blur, frame2_blur)
            # cv2.imwrite("diff.jpg", diff)
            # input("stop to view the diff")

            # סינון רעש דינמי
            mean_intensity = np.mean(diff)
            dynamic_threshold = base_threshold + int(mean_intensity * 0.5)

            _, binary_diff = cv2.threshold(
                diff, dynamic_threshold, 255, cv2.THRESH_BINARY)

            # החלקה זמנית
            current_motion = binary_diff.astype(np.float32)
            smoothed_motion = cv2.addWeighted(
                current_motion, 1 - temporal_smoothing,
                prev_motion, temporal_smoothing, 0
            )

            # עדכון המצטבר
            motion_accumulator = cv2.addWeighted(
                motion_accumulator, 0.7,
                smoothed_motion, 0.3, 0
            )

            prev_motion = smoothed_motion
            # for diff in 2 frames
            # cv2.imwrite("prev_motion.jpg", prev_motion)
            # input("stop to view the prev_motion")

        # for diff in 2 frames
        # cv2.imwrite("motion_accumulator.jpg", motion_accumulator)
        # input("stop to view the motion_accumulator")

        return motion_accumulator.astype(np.uint8)

    @staticmethod
    def detect_significant_movement(
        frames: List[np.ndarray],
        mask: np.ndarray = None,
        sensitivity: float = 1,
        min_area: int = 100,
        box_padding: int = 1,
        noise_threshold: int = 1,
        blur_strength: float = 0,
        temporal_smoothing: float = 0,
        mask_with_movement: bool = False
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        זיהוי תנועה משמעותית בפריימים עם פרמטרים נוספים לשליטה.

        Args:
            frames: רשימת פריימי קלט (לפחות 2)
            mask: מסכה בינארית אופציונלית להגבלת אזור הזיהוי
            sensitivity: רגישות זיהוי תנועה (0.0 עד 1.0)
            min_area: שטח מינימלי לזיהוי תנועה משמעותית
            box_padding: ריפוד סביב תיבות התנועה שזוהו
            noise_threshold: סף לסינון רעש (ערכים נמוכים יותר = רגישות גבוהה יותר)
            blur_strength: עוצמת הטשטוש (1.0 = רגיל, גבוה יותר = טשטוש חזק יותר)
            temporal_smoothing: החלקה זמנית בין פריימים (0.0 עד 1.0)

        Returns:
            Tuple של (has_motion, binary_mask, color_mask):
            - has_motion: בוליאני המציין אם זוהתה תנועה
            - binary_mask: מערך NumPy עם 0 ו-1
            - color_mask: מערך NumPy עם שחור (0,0,0) וירוק (0,255,0)
        """
        MaskService._validate_inputs(frames, mask)

        motion_mask = MaskService._create_motion_accumulator(
            frames, sensitivity, noise_threshold, blur_strength, temporal_smoothing
        )

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [
            contour for contour in contours
            if MaskService._is_valid_contour(contour, mask, min_area)
        ]

        binary_mask, color_mask = MaskService._create_bbox_masks(
            valid_contours, frames[0].shape, box_padding, True
        )
        # cv2.imwrite("color_mask1.jpg", color_mask)
        # input("stop to view the color_mask:")

        # cv2.imwrite('test.jpg', color_mask)
        if mask_with_movement:
            if isinstance(mask, np.ndarray):
                binary_mask = cv2.bitwise_and(mask, binary_mask)
        else:
            binary_mask = mask

        return bool(valid_contours), binary_mask, color_mask

    @staticmethod
    def print_results(detections: List[Detection] | List[List[Detection]], shape: tuple[int] = (10, 16, 16, 40)):
        """
        Prints the results of object detections in a formatted table.

        Args:
            detections: A list of detections or a list of lists of detections.
            shape: Column widths for the table.
        """
        if not detections or (isinstance(detections[0], list) and not any(detections)):
            print("\n\n\033[91m   - No detections found -   \033[0m\n\n")
            return

        if not isinstance(detections[0], list):
            detections = [detections]

        width = shutil.get_terminal_size().columns - 4
        B, F, S, L = shape
        B = max(1, (width // 2) - ((F + S + L) // 2) - 10)

        headers = f"{' ' * B}| {'Class Name':^{F}}|{'Confidence':^{S}}|{'bbox':^{L}}|"
        divider = f"{' ' * B}•-{'-' * F}•{'-' * S}•{'-' * L}•"

        print("\n\n" + divider)
        print(headers)
        print(divider)
        object_detected = defaultdict(lambda: "❓", {
            "car": "🚗",
            "person": "🚶"
        })

        for i, image_detections in enumerate(detections, 1):
            print(f"{' ' * B}| {f'image {i}':<{F}}|{' ' * S}|{' ' * L}|")

            for j, det in enumerate(image_detections, 1):
                print(f"{' ' * B}| {f'   {j}. {object_detected[det.class_name]} {det.class_name}':<{F-1}}|{f'{det.confidence}':^{S}}|{f'{det.bbox}':^{L}}|")

            print(divider)

        print()

    # def visualize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    #     """
    #     Converts a binary motion mask (0s and 1s) into a color image:
    #     - Black (0,0,0) for no motion (0)
    #     - Green (0,255,0) for motion (1)

    #     Args:
    #         binary_mask: NumPy array with values 0 and 1.

    #     Returns:
    #         Color image highlighting motion in green.
    #     """
    #     color_mask = np.zeros(
    #         (binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    #     color_mask[binary_mask == 1] = (0, 255, 0)
    #     return color_mask

    # @staticmethod
    # def accumulate_motion(prev_diff, current_diff, alpha=0.5):
    #     """
    #     Accumulates motion between two frames by blending their differences.

    #     Args:
    #         prev_diff: The previous frame's difference.
    #         current_diff: The current frame's difference.
    #         alpha: Weight for the previous frame's difference in the blend.

    #     Returns:
    #         Accumulated motion.
    #     """
    #     return cv2.addWeighted(prev_diff.astype(np.float32), alpha, current_diff.astype(np.float32), 1 - alpha, 0)

    # @staticmethod
    # def is_contour_in_mask(contour, mask):
    #     """
    #     Checks if a contour's center is inside the provided mask.

    #     Args:
    #         contour: The contour to check.
    #         mask: The mask to check against.

    #     Returns:
    #         True if the contour's center is inside the mask, False otherwise.
    #     """
    #     M = cv2.moments(contour)
    #     if M["m00"] == 0:
    #         return False
    #     cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    #     if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
    #         return mask[cy, cx] == 1
    #     return False

    # @staticmethod
    # def detect_significant_movement(frames: List[np.ndarray], mask: np.ndarray, sensitivity: float = 1.0, min_area: int = 500) -> bool:
    #     """
    #     Detects significant motion in a series of frames using a mask.

    #     Args:
    #         frames: List of frames to analyze (at least two).
    #         mask: Binary mask to apply.
    #         sensitivity: Motion sensitivity.
    #         min_area: Minimum area to consider significant motion.

    #     Returns:
    #         True if significant motion is detected, False otherwise.
    #     """

    #     if len(frames) < 2:
    #         raise ValueError("At least two frames are required.")

    #     frame_shape = frames[0].shape
    #     if not all(frame.shape == frame_shape for frame in frames):
    #         raise ValueError("All frames must have the same shape.")

    #     if mask.shape[:2] != frame_shape[:2]:
    #         raise ValueError("Mask shape must match frame shape.")

    #     threshold = 25 + int(sensitivity * 50)
    #     blur_kernel_size = max(int(15 * sensitivity), 3)
    #     min_area = int(min_area * (1 - sensitivity))

    #     motion_accumulator = np.zeros(frame_shape[:2], dtype=np.float32)

    #     for i in range(len(frames) - 1):
    #         diff = cv2.absdiff(frames[i], frames[i + 1])
    #         diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #         blurred_diff = cv2.GaussianBlur(
    #             diff_gray, (blur_kernel_size, blur_kernel_size), 0)
    #         _, binary_diff = cv2.threshold(
    #             blurred_diff, threshold, 255, cv2.THRESH_BINARY)

    #         motion_accumulator = MaskService.accumulate_motion(
    #             motion_accumulator, binary_diff)
    #     print(motion_accumulator)

    #     contours, _ = cv2.findContours(motion_accumulator.astype(
    #         np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     significant_contours = [
    #         cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    #     filtered_contours = [
    #         cnt for cnt in significant_contours if MaskService.is_contour_in_mask(cnt, mask)]
    #     if len(filtered_contours) < 1:
    #         return False, []
    #     motion_polygons, color_mask = MaskService.create_motion_mask(frame_shape=frames[0].shape, motion_polygons=[
    #         cnt.reshape(-1, 2) for cnt in filtered_contours])

    #     return len(filtered_contours) > 0, [motion_polygons, color_mask]

    # @staticmethod
    # def create_motion_mask(frame_shape: Tuple[int, int], motion_polygons: List[np.ndarray]) -> np.ndarray:
    #     """
    #     Creates a binary mask from detected motion polygons.

    #     Args:
    #         frame_shape: Tuple (height, width) of the frame.
    #         motion_polygons: List of NumPy arrays representing polygons of detected motion areas.

    #     Returns:
    #         Binary mask (NumPy array) where motion areas are 1, and others are 0.
    #     """
    #     mask = np.zeros(
    #         frame_shape[:2], dtype=np.uint8)
    #     cv2.fillPoly(mask, motion_polygons, 1)

    #     color_mask = np.zeros(
    #         (mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #     color_mask[mask == 1] = (0, 255, 0)

    #     return mask, color_mask

        # def _create_motion_accumulator(frames: List[np.ndarray], sensitivity: float) -> np.ndarray:
    #     """
    #     Create motion accumulator from consecutive frames.

    #     Args:
    #         frames: List of input frames
    #         sensitivity: Motion detection sensitivity

    #     Returns:
    #         Motion accumulator mask
    #     """
    #     threshold = 25 + int(sensitivity * 50)
    #     blur_kernel_size = max(int(15 * sensitivity), 3)
    #     motion_accumulator = np.zeros(frames[0].shape[:2], dtype=np.float32)

    #     for i in range(len(frames) - 1):
    #         diff = cv2.absdiff(frames[i], frames[i + 1])
    #         diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #         blurred_diff = cv2.GaussianBlur(
    #             diff_gray, (blur_kernel_size, blur_kernel_size), 0)
    #         _, binary_diff = cv2.threshold(
    #             blurred_diff, threshold, 255, cv2.THRESH_BINARY)

    #         motion_accumulator = cv2.addWeighted(
    #             motion_accumulator, 0.5,
    #             binary_diff.astype(np.float32), 0.5, 0)

    #     return motion_accumulator.astype(np.uint8)

#     @staticmethod
#     def detect_significant_movement(
#         frames: List[np.ndarray],
#         mask: np.ndarray = None,

#         sensitivity: float = 1.0,
#         min_area: int = 100,
#         box_padding: int = 5,
#         mash_with_movement: bool = True
#     ) -> Tuple[bool, np.ndarray, np.ndarray]:
#         """
#         Detect significant motion in frames and return motion status and masks.

#         Args:
#             frames: List of input frames (at least 2)
#             mask: Optional binary mask to limit detection area
#             sensitivity: Motion detection sensitivity (0.0 to 1.0)
#             min_area: Minimum area to consider as significant motion
#             box_padding: Padding to add around detected motion boxes

#         Returns:
#             Tuple of (has_motion, binary_mask, color_mask) where:
#             - has_motion: Boolean indicating if motion was detected
#             - binary_mask: NumPy array with 0s and 1s
#             - color_mask: NumPy array with black (0,0,0) and green (0,255,0)
#         """
#         # Validate inputs
#         MaskService._validate_inputs(frames, mask)

#         # Create motion accumulator
#         motion_mask = MaskService._create_motion_accumulator(
#             frames, sensitivity)

#         # Find contours
#         contours, _ = cv2.findContours(
#             motion_mask,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )

#         # Filter valid contours
#         valid_contours = [
#             contour for contour in contours
#             if MaskService._is_valid_contour(contour, mask, min_area)
#         ]

#         # Create masks from valid contours
#         binary_mask = MaskService._create_bbox_masks(
#             valid_contours, frames[0].shape, box_padding
#         )

#         # cv2.imwrite('test.jpg', test_color_mask)
#         if mash_with_movement:
#             if isinstance(mask, np.ndarray):
#                 binary_mask = cv2.bitwise_and(mask, binary_mask)
#         else:
#             binary_mask = mask

#         return bool(valid_contours), binary_mask
