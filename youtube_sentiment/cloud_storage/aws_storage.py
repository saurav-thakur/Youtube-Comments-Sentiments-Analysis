import boto3
import os,sys
from mypy_boto3_s3.service_resource import Bucket
from botocore.exceptions import ClientError
from pandas import DataFrame,read_csv
from typing import Union,List
from io import StringIO
import tempfile
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.configuration.aws_connection import S3Client
from youtube_sentiment.logger import logging
from youtube_sentiment.utils.utilities import load_keras_model

class SimpleStorageService:

    def __init__(self) -> None:
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self,bucket_name,s3_key)->bool:

        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]

            if len(file_objects) > 0:
                return True
            return False
        except Exception as e:
            raise YoutubeException(e,sys)
        
    @staticmethod
    def read_object(object_name:str,decode: bool=True,make_readable:bool = False) -> Union[StringIO,str]:

        logging.info("exited the read_object method of SimpleStorageService class")
        try:
            func = (lambda: object_name.get()["Body"].read().decode()
                    if decode is True
                    else object_name.get()["Body"].read())
            
            conv_func = lambda: StringIO(func()) if make_readable is True else False
            logging.info("exited the read_object method of SimpleStorageService class")

            return conv_func()

        except Exception as e:
            raise YoutubeException(e,sys)
        


    def get_bucket(self, bucket_name: str) -> Bucket:
            """
            Method Name :   get_bucket
            Description :   This method gets the bucket object based on the bucket_name

            Output      :   Bucket object is returned based on the bucket name
            On Failure  :   Write an exception log and then raise an exception

            Version     :   1.2
            Revisions   :   moved setup to cloud
            """
            logging.info("Entered the get_bucket method of S3Operations class")

            try:
                bucket = self.s3_resource.Bucket(bucket_name)
                logging.info("Exited the get_bucket method of S3Operations class")
                return bucket
            except Exception as e:
                raise YoutubeException(e, sys)

    def get_file_object( self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Method Name :   get_file_object
        Description :   This method gets the file object from bucket_name bucket based on filename

        Output      :   list of objects or object is returned based on filename
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the get_file_object method of S3Operations class")

        try:
            bucket = self.get_bucket(bucket_name)

            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(file_objects)
            logging.info("Exited the get_file_object method of S3Operations class")

            return file_objs

        except Exception as e:
            raise YoutubeException(e, sys)

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Method Name :   load_model
        Description :   This method loads the model_name model from bucket_name bucket with kwargs

        Output      :   Keras model object
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.3
        Revisions   :   Updated to support Keras 3 file formats
        """
        
        """
         the usage of tempfile.NamedTemporaryFile:

        tempfile.NamedTemporaryFile(delete=False, suffix='.keras'):

        This creates a temporary file with a unique name.
        delete=False: This means the file won't be automatically deleted when closed. We need this because we're going to use the file after the with block ends.
        suffix='.keras': This adds the '.keras' extension to the temporary file, which can be helpful for some operations that rely on file extensions.


        temp_file.write(file_object.get()['Body'].read()):

        This writes the content of the S3 object to the temporary file.


        temp_file_path = temp_file.name:

        This gets the path of the temporary file, which we'll use to load the model.


        model = keras.models.load_model(temp_file_path):

        This loads the Keras model from the temporary file.


        os.unlink(temp_file_path):

        This deletes the temporary file after we've loaded the model.



        Using a temporary file like this has several advantages:

        It allows us to work with file-based APIs (like keras.models.load_model) that expect a file path, even when our data is coming from a non-file source like S3.
        It's secure, as the temporary file is created with permissions that only allow the current user to read it.
        It ensures we don't leave unnecessary files on the disk, as we delete it immediately after use.
        """
        logging.info("Entered the load_model method of S3Operations class")

        try:
            model_file = model_name if model_dir is None else f"{model_dir}/{model_name}"
            file_object = self.get_file_object(model_file, bucket_name)
            
            if file_object is None:
                raise FileNotFoundError(f"Model file {model_file} not found in bucket {bucket_name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
                temp_file.write(file_object.get()['Body'].read())
                temp_file_path = temp_file.name

            model = load_keras_model(temp_file_path)
            os.unlink(temp_file_path)  # Remove the temporary file

            logging.info("Exited the load_model method of S3Operations class")
            return model

        except Exception as e:
            raise YoutubeException(e, sys)
    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Method Name :   create_folder
        Description :   This method creates a folder_name folder in bucket_name bucket

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the create_folder method of S3Operations class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            else:
                pass
            logging.info("Exited the create_folder method of S3Operations class")

    def upload_file(self, from_filename: str, to_filename: str,  bucket_name: str,  remove: bool = True):
        """
        Method Name :   upload_file
        Description :   This method uploads the from_filename file to bucket_name bucket with to_filename as bucket filename

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the upload_file method of S3Operations class")

        try:
            logging.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            logging.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            if remove is True:
                os.remove(from_filename)

                logging.info(f"Remove is set to {remove}, deleted the file")

            else:
                logging.info(f"Remove is set to {remove}, not deleted the file")

            logging.info("Exited the upload_file method of S3Operations class")

        except Exception as e:
            raise YoutubeException(e, sys)

    def upload_df_as_csv(self,data_frame: DataFrame,local_filename: str, bucket_filename: str,bucket_name: str,) -> None:
        """
        Method Name :   upload_df_as_csv
        Description :   This method uploads the dataframe to bucket_filename csv file in bucket_name bucket

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the upload_df_as_csv method of S3Operations class")

        try:
            data_frame.to_csv(local_filename, index=None, header=True)

            self.upload_file(local_filename, bucket_filename, bucket_name)

            logging.info("Exited the upload_df_as_csv method of S3Operations class")

        except Exception as e:
            raise YoutubeException(e, sys)

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method gets the dataframe from the object_name object

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logging.info("Exited the get_df_from_object method of S3Operations class")
            return df
        except Exception as e:
            raise YoutubeException(e, sys)

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method gets the dataframe from the object_name object

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the read_csv method of S3Operations class")

        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of S3Operations class")
            return df
        except Exception as e:
            raise YoutubeException(e, sys)