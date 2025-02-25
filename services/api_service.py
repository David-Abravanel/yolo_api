# import cv2
# import aiohttp
# import numpy as np
# from typing import Optional


# class S3Service:
#     @staticmethod
#     async def fetch_image(url: str) -> Optional[np.ndarray] | False:
#         try:
#             # TODO: revers to S3 client like in the file -> test/a.py
#             async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
#                 async with session.get(url) as response:
#                     if response.status != 200:
#                         content_type = response.headers.get(
#                             'Content-Type', '').lower()
#                         if 'xml' in content_type:
#                             return False
#                         raise ValueError(f"Failed to fetch image: {response.status}, {await response.text()}")
#                     image_data = await response.read()

#                     nparr = np.frombuffer(image_data, np.uint8)

#                     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                     if img is None:
#                         raise ValueError("Failed to decode image data.")
#                     return img

#         except aiohttp.ClientError as e:
#             raise ValueError(f"Network error occurred: {str(e)}")
#         except Exception as e:
#             raise ValueError(f"Unexpected error: {str(e)}")

import aioboto3
import cv2
import numpy as np
from typing import Optional


class S3Service:
    _instance = None

    def __new__(cls, region: str, source_bucket: str, source_folder: str):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.session = aioboto3.Session(region_name=region)
            cls.bucket_name = source_bucket
            cls.images_folder = source_folder
            cls._instance.S3 = None
        return cls._instance

    async def initialize(self):
        self.S3 = await self.session.client('s3').__aenter__()

    async def _get_s3_client(self):
        """וודא שה- S3 client פתוח ונשאר פתוח לכל הקריאות"""
        if self.S3 is None:
            self.S3 = await self.session.client('s3').__aenter__()
        return self.S3

    async def fetch_image(self, key: str) -> Optional[np.ndarray] | False:
        try:
            S3 = await self._get_s3_client()
            response = await S3.get_object(Bucket=self.bucket_name, Key=f'{self.images_folder}/{key}')
            image_data = await response['Body'].read()
            # 
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image data.")
            return img

        except aioboto3.exceptions.S3UploadFailedError as e:
            raise ValueError(f"Failed to fetch image from S3: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")

    async def upload_image(self, key: str, image: np.ndarray) -> bool:
        try:
            S3 = await self._get_s3_client()
            _, encoded_image = cv2.imencode('.jpg', image)
            image_data = encoded_image.tobytes()
            await S3.put_object(Bucket=self.bucket_name, Key=f'{self.images_folder}/{key}', Body=image_data, ContentType='image/jpeg')
            print("🖼️⬆️ mask color uploaded")
            return True

        except aioboto3.exceptions.S3UploadFailedError as e:
            raise ValueError(f"Failed to upload image to S3: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")
