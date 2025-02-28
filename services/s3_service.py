import aioboto3
import cv2
import numpy as np
from typing import Optional


class S3Service:
    _instance = None

    def __new__(
        cls,
        region: str,
        source_bucket: str,
        source_folder: str,
        dest_bucket: str,
        dest_folder: str
    ):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.region = region
            cls._instance.session = aioboto3.Session(region_name=region)
            cls._instance.temp_bucket = source_bucket
            cls._instance.temp_snaps_folder = source_folder
            cls._instance.backend_bucket = dest_bucket
            cls._instance.backend_snaps_folder = dest_folder
            cls._instance.S3 = None
        return cls._instance

    async def initialize(self):
        self.S3 = await self.session.client(
            "s3",
            region_name="il-central-1",
            endpoint_url="https://s3.il-central-1.amazonaws.com"
        ).__aenter__()

    async def _get_s3_client(self, fresh: bool = False):
        """Ensures the S3 client is initialized with the correct endpoint"""
        if fresh or self.S3 is None:
            if self.S3 is not None:
                try:
                    await self.S3.__aexit__(None, None, None)
                except Exception:
                    pass
                self.S3 = None

            self.S3 = await self.session.client(
                "s3",
                region_name="il-central-1",
                endpoint_url="https://s3.il-central-1.amazonaws.com"
            ).__aenter__()

        return self.S3

    async def fetch_image(self, key: str) -> Optional[np.ndarray] | False:
        try:
            S3 = await self._get_s3_client()
            response = await S3.get_object(Bucket=self.temp_bucket, Key=f'{self.temp_snaps_folder}/{key}')
            image_data = await response['Body'].read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image data.")
            return img

        except aioboto3.exceptions.S3UploadFailedError as e:
            raise ValueError(f"Failed to fetch image from S3: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")

    async def upload_image(self, key: str, image: np.ndarray, bucket: Optional[str] = None, folder: Optional[str] = None) -> bool:
        try:
            S3 = await self._get_s3_client()
            _, encoded_image = cv2.imencode('.jpg', image)
            image_data = encoded_image.tobytes()
            bucket = bucket or self.backend_bucket
            folder = folder or self.backend_snaps_folder
            await S3.put_object(Bucket=bucket, Key=f'{folder}/{key}', Body=image_data, ContentType='image/jpeg')
            print("🖼️⬆️ mask color uploaded")
            return True

        except aioboto3.exceptions.S3UploadFailedError as e:
            raise ValueError(f"Failed to upload image to S3: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")

    async def generate_url(self, key: str, bucket: Optional[str] = None, expiration: int = 3600, folder: Optional[str] = None) -> str:
        """Generates a presigned URL for temporary file access."""
        s3 = await self._get_s3_client()
        bucket = bucket or self.backend_bucket
        folder = folder or self.backend_snaps_folder
        full_key = f'{folder}/{key}'
        # print(bucket, "/", full_key)
        try:
            url = await s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': full_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(e)
            raise ValueError(f"Failed to generate presigned URL: {str(e)}")

    async def move_pictures(self, key: str) -> list[bool | Exception]:
        """Move multiple objects from the current bucket to another bucket/folder."""

        S3 = await self._get_s3_client()
        copy_source = {'Bucket': self.temp_bucket,
                       'Key': f'{self.temp_snaps_folder}/{key}'}
        dest_key = f'{self.backend_snaps_folder}/{key}'

        try:
            await S3.copy_object(Bucket=self.backend_bucket, Key=dest_key, CopySource=copy_source)
            return True
        except Exception as e:
            print(e)
            return e
