__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void aco_path_planning_3d
(
  DTYPE_IMAGE_OUT_3D dstAnts,
  DTYPE_IMAGE_IN_3D srcAnts,
  DTYPE_IMAGE_IN_3D srcFitness,
  DTYPE_IMAGE_IN_3D srcPheromone,
  DTYPE_IMAGE_IN_3D srcRandom,
  float alpha,
  float beta
)
{
  const int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  const int4 coord = (int4)(i,j,k,0);

  // if there is no ant, leave
  float ant = READ_IMAGE(srcAnts, sampler, coord).x;
  if (ant == 0) {
    return;
  }

  const int4   c = (int4)  (1, 1, 1, 0 );

  float sum = 0;

  for (int x = -c.x; x <= c.x; x++) {
    for (int y = -c.y; y <= c.y; y++) {
      for (int z = -c.z; z <= c.z; z++) {
        sum += pow((float)READ_IMAGE(srcPheromone,sampler,coord+(int4)(x,y,z,0)).x, alpha) *
               pow((float)READ_IMAGE(srcFitness,sampler,coord+(int4)(x,y,z,0)).x, beta);
      }
    }
  }

  float random = (float)READ_IMAGE(srcRandom,sampler,coord).x;
  float threshold = random * sum / 255;

  sum = 0;
  for (int x = -c.x; x <= c.x; x++) {
    for (int y = -c.y; y <= c.y; y++) {
      for (int z = -c.z; z <= c.z; z++) {
        sum += pow((float)READ_IMAGE(srcPheromone,sampler,coord+(int4)(x,y,z,0)).x, alpha) *
               pow((float)READ_IMAGE(srcFitness,sampler,coord+(int4)(x,y,z,0)).x, beta);
        if (sum > threshold) {
          WRITE_IMAGE(dstAnts,coord+(int4)(x,y,z,0),(DTYPE_OUT)ant);
          return;
        }
      }
    }
  }
}

__kernel void aco_seed_descendants_3d
(
  DTYPE_IMAGE_OUT_3D dstAnts,
  DTYPE_IMAGE_IN_3D srcAnts,
  DTYPE_IMAGE_IN_3D srcPheromone,
  DTYPE_IMAGE_IN_3D srcRandom,
  float pheromoneThreshold
) {

  const int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  const int4 coord = (int4)(i,j,k,0);

  const int4   c = (int4)  (1, 1, 1, 0 );

  // if there is an ant already, leave
    float ant = READ_IMAGE(srcAnts, sampler, coord).x;
    if (ant > 0) {
      // copy pre-existing ants
      WRITE_IMAGE(dstAnts,coord, (DTYPE_OUT)ant);
      return;
    }

    float antsInNeighborhood = 0;
    for (int x = -c.x; x <= c.x; x++) {
      for (int y = -c.y; y <= c.y; y++) {
        for (int z = -c.z; z <= c.z; z++) {

          // if there is an ant and its at a location where there is enough pheromone
          if (READ_IMAGE(srcAnts,sampler,coord+(int4)(x,y,z,0)).x > 0 &&
              READ_IMAGE(srcPheromone,sampler,coord+(int4)(x,y,z,0)).x > pheromoneThreshold) {
            antsInNeighborhood ++;
          }
        }
      }
    }

    if (antsInNeighborhood * 255 / 27 > READ_IMAGE(srcRandom,sampler,coord).x) {
      WRITE_IMAGE(dstAnts,coord,(DTYPE_OUT)1.0);
    } else {
      WRITE_IMAGE(dstAnts,coord,(DTYPE_OUT)0.0);
    }

}




