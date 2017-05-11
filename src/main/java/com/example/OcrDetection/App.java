package com.example.OcrDetection;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Stack;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.SVM;

/**
 * Hello world!
 *
 */
public class App 
{
	static{System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
	
	static int width;
	static int height;
	static int px_count;
	static int total_images;
    public static void main( String[] args ) throws FileNotFoundException
    {
    	ReadMNISTData();
    	//Mat test_image=Imgcodecs.imread("digits.png",0);
//    	System.out.println("Output image result is :"+FindMatch(test_image));
        
    }
    public static void hcrDetector(Mat img){
    	
    	
    	
    }
    public static void ReadMNISTData() throws FileNotFoundException{
    	String images_path="train-images.idx3-ubyte";
    	File mnist_images_file = new File(images_path);
    	FileInputStream images_reader = new	FileInputStream(mnist_images_file);
    	Mat training_images = null;
    	
    	try {
    		byte [] header = new byte[16];
			images_reader.read(header, 0, 16);
			//Combining the bytes to form an integer
			ByteBuffer temp = ByteBuffer.wrap(header, 4, 12);
			 total_images = temp.getInt();
			 width = temp.getInt();
			 height = temp.getInt();
			//Total number of pixels in each image
			 px_count = width * height;
			training_images = new Mat(total_images, px_count,
			CvType.CV_8U);
			//images_data = new byte[total_images][px_count];
			//Read each image and store it in an array.
			for (int i = 0 ; i < total_images ; i++)
			{
			byte[] image = new byte[px_count];
			images_reader.read(image, 0, px_count);
			training_images.put(i,0,image);
			}
			training_images.convertTo(training_images,CvType.CV_32FC1);
			images_reader.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
    	//Read Labels
    	Mat training_labels = null;
    	byte []labels_data = new byte[total_images];
    	String labels_path="train-labels.idx1-ubyte";
    
    	File mnist_labels_file = new File(labels_path);
    	FileInputStream labels_reader = new FileInputStream(mnist_labels_file);
    	try{
    	training_labels = new Mat(total_images, 1,
    	CvType.CV_8U);
    	Mat temp_labels = new Mat(1, total_images,
    	CvType.CV_8U);
    	byte[] header = new byte[8];
    	//Read the header
    	labels_reader.read(header, 0, 8);
    	//Read all the labels at once
    	labels_reader.read(labels_data,0,total_images);
    	temp_labels.put(0,0, labels_data);
    	//Take a transpose of the image
    	Core.transpose(temp_labels, training_labels);
    	training_labels.convertTo(training_labels,
    	CvType.CV_32FC1);
    	labels_reader.close();
    	}
    	catch (IOException e)
    	{
    	System.out.println("MNIST Read Error:"+ "" + e.getMessage());
    	}
    	
    	SVM knn =  SVM.create();
		
		System.out.println(knn.isTrained());
		knn.train(training_images, 10,training_labels);
    	
    	System.out.println(knn.isTrained());
	} 
    
   public static Mat FindMatch(Mat test_image) throws FileNotFoundException
    {
	  ReadMNISTData();
    //Dilate the image
    Imgproc.dilate(test_image, test_image,
    Imgproc.getStructuringElement(Imgproc.CV_SHAPE_CROSS,
    new Size(3,3)));
    //Resize the image to match it with the sample image size
    Imgproc.resize(test_image, test_image, new
    Size(width, height));
    //Convert the image to grayscale
    Imgproc.cvtColor(test_image, test_image,
    Imgproc.COLOR_RGB2GRAY);
    //Adaptive Threshold
    Imgproc.adaptiveThreshold(test_image,test_image,
    255,Imgproc.ADAPTIVE_THRESH_MEAN_C,
    Imgproc.THRESH_BINARY_INV,15, 2);
    Mat test = new Mat(1, test_image.rows() *
    test_image.cols(), CvType.CV_32FC1);
    int count = 0;
    for(int i = 0 ; i < test_image.rows(); i++)
    {
    for(int j = 0 ; j < test_image.cols(); j++) {
    test.put(0, count, test_image.get(i, j)[0]);
    count++;
    }
    }
    Mat results = new Mat(1, 1, CvType.CV_8U);
    
    
    KNearest knn =KNearest.create();
    knn.findNearest(test, 10, results, new Mat(), new Mat());
    System.out.println("Result:"+ "" + results.get(0,0)[0]);
    return results;
    }
}
