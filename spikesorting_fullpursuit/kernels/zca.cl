/**
 * @file
 * @author David J. Herzfeld <herzfeldd@gmail.com>
 *
 * @section LICENSE
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTIBILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details at <http://www.gnu.org/licenses/>
 *
 * @section DESCRIPTION
 */

 /* Allow customization of the memory type for the GPU */
 #ifndef voltage_type
 #define voltage_type float
 #endif

 /**
  * Kernel used to compute the ZCA whitening across channels
  *
  * This kernel takes a matrix of voltages of dimensions (num_channels x num_timepoints)
  * and then compute the ZCA transformed matrix. Each worker computes the
  * transformation of a single row/column. Therefore, this function should be
  * called with the number of workers > the number of elements in voltage.
  *
  * Note that voltage should be packed such that all timepoints for channel 1
  * occur first, followed by all timepoints for channel 2, etc. In addition,
  * voltage and voltage_out must be unique buffers - the matrix multiplication
  * cannot be performed in-place.
  *
  * Parameters:
  *  voltage: A voltage_type vector with dimensions num_channels x num_timepoints
  *  num_channels: The number of channels (rows in voltage)
  *  num_timepoints: The number of columns in voltage.
  *  Z: A num_channelsxnum_channels matrix that describes the matrix multiplication required
  *   to compute the ultimate transform.
  *  voltage_out [out]: An output matrix the same size as voltage where we will
  *   place the final computed matrix.
  */
  __kernel void zca_whiten(
      __global const voltage_type * restrict voltage,
      const unsigned int num_channels,
      const unsigned int num_timepoints,
      __global const float * restrict Z,
      __global voltage_type * restrict voltage_out)
{
    const size_t id = get_global_id(0);
    unsigned int i;
    if (id >= num_timepoints * num_channels)
    {
        return; /* Nothing to do */
    }
    const unsigned int my_row = id / num_timepoints;
    const unsigned int my_column = id - (my_row * num_timepoints);

    float output_voltage = 0;
    for (i = 0; i < num_channels; i++)
    {
        output_voltage = output_voltage + Z[my_row + i*num_channels] * (float) voltage[my_column + i*num_timepoints];
        /* NOTE: Because Z is a symmetric var-covar matrix, this line is equivalent to: */
        /* output_voltage = output_voltage + Z[my_row * num_channels + i] * voltage[my_column + i*num_timepoints]; */
    }
    voltage_out[id] = (voltage_type) output_voltage;

}
