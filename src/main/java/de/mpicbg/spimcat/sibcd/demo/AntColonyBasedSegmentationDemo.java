package de.mpicbg.spimcat.sibcd.demo;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.kernels.Kernels;
import clearcl.util.ElapsedTime;
import de.mpicbg.spimcat.sibcd.AntColonyBasedSegmentation;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.io.IOException;

/**
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * April 2018
 */
public class AntColonyBasedSegmentationDemo
{

  public static void main (String... args) throws IOException
  {
    new ImageJ();
    ClearCLIJ clij = ClearCLIJ.getInstance("HD");
    ElapsedTime.sStandardOutput = true;

    String file = "src/main/resources/droso_crop.tif";

    ImagePlus imp = IJ.openImage(file);
    imp.show();

    //for (int i = 0; i < 1000; i++) {
      ElapsedTime.measure("the whole thing ", () -> {

        ClearCLImage input = clij.converter(imp).getClearCLImage();
        ClearCLImage output = clij.createCLImage(new long[]{input.getWidth()/2, input.getHeight()/2, input.getDepth()}, input.getChannelDataType());

        AntColonyBasedSegmentation acbs = new AntColonyBasedSegmentation(clij, input, output);
        acbs.initialize();
        for (int i = 0; i < 10; i++) {
          acbs.exec();
        }
        acbs.cleanup();

        System.out.println("spots: " + Kernels.sumPixels(clij, output));

        input.close();
        output.close();

      });
    //}
  }

}
