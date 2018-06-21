package de.mpicbg.spimcat.sibcd;

import clearcl.ClearCLImage;
import clearcl.enums.ImageChannelDataType;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.kernels.Kernels;
import clearcl.imagej.utilities.ImageTypeConverter;
import de.mpicbg.spimcat.sibcd.kernels.ACO;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;

import java.util.HashMap;
import java.util.Random;

public class AntColonyBasedSegmentation {

    private float initialAntSeedProbability = 0.04f; // seed ants on 4 percent of pixels at the beginning
    private float alpha = 0.5f;
    private float beta = 0.5f;


    private ClearCLIJ clij;

    private ClearCLImage input;
    private ClearCLImage output;

    private ClearCLImage pheromone;
    private ClearCLImage temp1;
    private ClearCLImage temp2;

    private ClearCLImage ants;

    private Random randomGenerator = new Random(System.currentTimeMillis());
    private ClearCLImage random;


    public AntColonyBasedSegmentation(ClearCLIJ clij, ClearCLImage input, ClearCLImage output) {
        this.clij = clij;

        // initialisation
        this.input = input;
        this.output = output;
        pheromone = clij.createCLImage(input.getDimensions(), ImageChannelDataType.UnsignedInt8);
        temp1 = clij.createCLImage(input.getDimensions(), ImageChannelDataType.UnsignedInt8);
        temp2 = clij.createCLImage(input.getDimensions(), ImageChannelDataType.UnsignedInt8);

        ants = clij.createCLImage(input.getDimensions(), ImageChannelDataType.UnsignedInt8);
        random = clij.createCLImage(input.getDimensions(), ImageChannelDataType.UnsignedInt8);
    }

    public void initialize() {

        // distribute ants, use a randomized image with values between 0 and 255 and
        // threshold it with percentage times 255
        randomize(clij, random, 0, 255);
        Kernels.threshold(clij, random, ants, 255 * (1.0f - initialAntSeedProbability));

        Kernels.set(clij, pheromone, 1);
    }

    public void exec()
    {
        // ant motion
        randomize(clij, random, 0, 255);

        HashMap<String, Object> parameters = new HashMap<String, Object>();
        parameters.put("dstAnts", temp1);
        parameters.put("srcAnts", ants);
        parameters.put("srcFitness", input);
        parameters.put("srcPheromone", pheromone);
        parameters.put("srcRandom", random);
        parameters.put("alpha", alpha);
        parameters.put("beta", beta);
        clij.execute(ACO.class, "aco.cl", "aco_path_planning_3d", parameters);

        clij.show(ants, "ants before");
        clij.show(temp1, "ants after");

        // measure mean pheromone among ants

        // seed descendants

        // update pheromone

        // evaporate pheromone

    }

    public void cleanup() {

    }

    public void randomize(ClearCLIJ clij, ClearCLImage image, float minValue, float maxValue) {
        Img<FloatType> img = ArrayImgs.floats(image.getDimensions());

        Cursor<FloatType> cursor = img.cursor();
        while (cursor.hasNext()) {
            cursor.next().set(randomGenerator.nextFloat()* (maxValue - minValue) + minValue);
        }

        ImageTypeConverter.copyRandomAccessibleIntervalToClearCLImage(img, image);
    }

}