#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <lstm.h>
#include <unistd.h>

#ifdef SIEKNET_USE_GPU
#include <opencl_utils.h>
#endif

int ALLOCS = 0;
int FREES  = 0;

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;
#define PRINTLIST(name, len) printf("printing %s: {", #name); for(int xyz = 0; xyz < len; xyz++){printf("%f", name[xyz]); if(xyz < len-1) printf(", "); else printf("};\n");}

float uniform(float minimum, float maximum){
  float center = minimum + (maximum - minimum)/2;
  float max_mag = maximum - center;
  if(rand()&1)
    return center + ((((float)rand())/RAND_MAX)) * max_mag;
  else
    return center - ((((float)rand())/RAND_MAX)) * max_mag;
}

int assert_equal(float *v1, float *v2, size_t len){
  int equal = 1;
  for(int i = 0; i < len; i++){
    float diff = v1[i] - v2[i];
    if(diff < 0) diff*=-1;
    if(diff > 0.0001){
      equal = 0;
      //printf("%f is not %f (idx %d)\n", v1[i], v2[i], i);
    }else{
      //printf("%f is %f (idx %d)\n", v1[i], v2[i], i);
    }
  }
  return equal;
}

#ifndef SIEKNET_USE_GPU

float *retrieve_array(float *arr, size_t len){
  return arr;
}
void assign_array(float *arr, float *dest, size_t len){
  for(int i = 0; i < len; i++)
    dest[i] = arr[i];
}
void dispose_array(float *arr){}

#else

float *retrieve_array(cl_mem arr, size_t len){
  float *tmp = (float*)malloc(sizeof(float)*len);
  memset(tmp, '\0', len*sizeof(float)); 
  check_error(clEnqueueReadBuffer(get_opencl_queue0(), arr, 1, 0, sizeof(float) * len, tmp, 0, NULL, NULL), "error reading from gpu (retrieve_array)");
  check_error(clFinish(get_opencl_queue0()), "waiting on queue to finish");
  ALLOCS++;
  return tmp;
}
void assign_array(float *arr, cl_mem dest, size_t len){
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), dest, 1, 0, sizeof(float) * len, arr, 0, NULL, NULL), "enqueuing cost gradient");
  check_error(clFinish(get_opencl_queue0()), "waiting on queue to finish");
}
void dispose_array(float *arr){
  FREES++;
  free(arr);
}

#endif
int main(){
  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_  __/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/     \n");
  printf("																					 \n");
  printf("preparing to run lstm unit test suite...\n");
  sleep(1);

  srand(1);

  /* LSTM tests */
  {
    printf("\n******** TESTING LSTM FUNCTIONALITY ********\n\n");
    sleep(1);
    float x1[] = {0.34, 1.00, 0.00, 0.25, 0.89};
    float x2[] = {0.99, 0.00, 1.00, 0.59, 0.89};
    float x3[] = {0.13, 0.66, 0.72, 0.01, 0.89};
    float y1[] = {1.00, 0.00, 0.25, 0.89};
    float y2[] = {0.09, 0.90, 0.50, 0.12};
    float y3[] = {0.66, 0.72, 0.03, 0.45};
    float l1_t1[] = {-0.0433, 0.0689, 0.0500, 0.1117, 0.0244, 0.0063};
    float l2_t1[] = {-0.0567, -0.0623, -0.0267, -0.0024, -0.0676, -0.0161, 0.0082};
    float l3_t1[] = {0.3191, 0.1995, 0.2757, 0.2058};
    float l1_t2[] = {0.1176, 0.0961, 0.0514, 0.2174, 0.0061, -0.0248};
    float l2_t2[] = {-0.0809, -0.0840, -0.0480, -0.0179, -0.1166, -0.0434, 0.0161};
    float l3_t2[] = {0.3249, 0.1953, 0.2711, 0.2087};
    float l1_t3[] = {0.0516, 0.0855, 0.0790, 0.2063, 0.0369, 0.0437};
    float l2_t3[] = {-0.0904, -0.0996, -0.0575, -0.0241, -0.1381, -0.0472, 0.0117};
    float l3_t3[] = {0.3279, 0.1937, 0.2681, 0.2103};
    float p[]  = {-0.2777, 0.2252, -0.1606, 0.2374, 0.2073, 0.1380, -0.2246, 0.1157, -0.2030, 0.2436, 0.2468, -0.3070, 0.2177, 0.1775, -0.0868, 0.2287, 0.0371, -0.1922, 0.2222, -0.2586, -0.1099, 0.3281, 0.2758, -0.0627, -0.3195, 0.2550, -0.0404, 0.1279, -0.3144, 0.2873, 0.2646, 0.0991, 0.3032, -0.1660, 0.0543, 0.2882, 0.1546, 0.1653, -0.0601, 0.2425, -0.2010, -0.2794, 0.0594, -0.1662, -0.0461, 0.1083, -0.3028, -0.2789, 0.1654, 0.1314, 0.2030, 0.0504, -0.0359, -0.1212, 0.1105, 0.1424, 0.1945, -0.2196, -0.1625, 0.2943, -0.1726, 0.1859, 0.2766, -0.0814, 0.2431, -0.3283, 0.2992, -0.1369, 0.2611, -0.0979, 0.2885, 0.0164, -0.3288, -0.0703, -0.2884, 0.0332, -0.1006, 0.2697, 0.0172, -0.1526, 0.2307, -0.0397, 0.1929, 0.1983, 0.1014, 0.1589, -0.2032, 0.2063, 0.0779, 0.0234, -0.3079, 0.1606, -0.2756, 0.1191, -0.1148, -0.2197, 0.0858, -0.2087, 0.1027, 0.0660, -0.0366, 0.2608, -0.0668, 0.1052, -0.0770, 0.1776, 0.1851, -0.1267, -0.1018, 0.0868, -0.1841, -0.2287, 0.2326, -0.2153, 0.1777, 0.2061, -0.1728, -0.1207, 0.2673, -0.0510, -0.0212, 0.0625, -0.2334, -0.0004, 0.1017, -0.2185, 0.0604, 0.2224, 0.2178, -0.2950, 0.0524, 0.2763, -0.3014, -0.1315, -0.2896, -0.2463, -0.0777, 0.1838, 0.1648, -0.3130, 0.2714, 0.2191, 0.3120, 0.2914, 0.2126, 0.2585, -0.0876, 0.1569, 0.2652, 0.2910, -0.1375, -0.1990, 0.1795, -0.1930, 0.2336, -0.2775, -0.0375, 0.1701, -0.2715, -0.2126, 0.0480, 0.0823, 0.0058, -0.1912, 0.1942, -0.1591, -0.0196, 0.2136, -0.0742, 0.2101, -0.2460, -0.1462, 0.3176, 0.2997, -0.1112, -0.0730, 0.3166, 0.2948, -0.0613, -0.2601, 0.3016, -0.2539, -0.1106, -0.1869, 0.2074, -0.1593, 0.2180, 0.0702, -0.2885, -0.1245, -0.2155, -0.2255, -0.0196, -0.1879, 0.0063, 0.0030, -0.2005, 0.2957, 0.0577, -0.1626, 0.2130, 0.0518, -0.2941, 0.2998, -0.1586, -0.3155, 0.2886, -0.2396, -0.0614, 0.0557, -0.2087, 0.0422, 0.3156, -0.0536, -0.1539, -0.3051, 0.2047, -0.1312, 0.1455, 0.0814, 0.1887, 0.0116, 0.2713, 0.1188, 0.0119, -0.0670, -0.2333, 0.2964, -0.1138, -0.2556, -0.2329, -0.3014, 0.2508, -0.2698, -0.1970, -0.0338, -0.0814, 0.1964, -0.2965, 0.0187, 0.1564, -0.1957, 0.1949, 0.2718, -0.1755, -0.2431, -0.0881, -0.1794, 0.3105, -0.0686, 0.1337, -0.2266, 0.1478, 0.2344, -0.0715, -0.0860, 0.0054, -0.2841, 0.1072, 0.1756, 0.0834, -0.0244, 0.2966, -0.1770, 0.3197, 0.1147, 0.0146, -0.0017, 0.0976, 0.2734, 0.1558, 0.0637, -0.3078, 0.0281, -0.2371, 0.0310, 0.2892, 0.1586, -0.2592, 0.1929, 0.2479, 0.0321, -0.3155, -0.0916, -0.2699, 0.2489, 0.0578, -0.2536, -0.3028, -0.1054, 0.1409, 0.2018, 0.1853, 0.0251, 0.0423, -0.0353, -0.1797, -0.2062, 0.0964, 0.2371, -0.0118, 0.1232, -0.1397, 0.0951, -0.0017, -0.0390, -0.0260, -0.2425, 0.2809, -0.0193, -0.1627, 0.2863, -0.1197, -0.1970, -0.1665, -0.2538, -0.2844, -0.2436, -0.1179, -0.2493, 0.2306, -0.2621, 0.2205, -0.1593, -0.1550, 0.2073, -0.2333, -0.0429, 0.0605, -0.2606, -0.0568, 0.0966, 0.2742, 0.1481, -0.0616, 0.2030, 0.0685, -0.2402, 0.0439, 0.1731, -0.2695, -0.2080, 0.2569, 0.2386, -0.1822, 0.2357, -0.0877, -0.2716, -0.0552, 0.1367, -0.1998, 0.1005, 0.1875, -0.2487, 0.1231, -0.1217, -0.1851, 0.2782, 0.2720, 0.1422, -0.0902, 0.0759, 0.0929, -0.2313, -0.1499, 0.2651, -0.0905, -0.0009, 0.0891, -0.2007, -0.0541, 0.2962, 0.0349, -0.0640, -0.1654, 0.1438, -0.2754, -0.1251, -0.2586, -0.2079, -0.1071, 0.1411, 0.2213, -0.1670, -0.0829, -0.2140, 0.0729, 0.0676, 0.1587, 0.1060, -0.0291, -0.2713, 0.1609, 0.0556, -0.1355, 0.2108, -0.1192, 0.1316, 0.1853, 0.0337, -0.0492, 0.0967, -0.1406, -0.2299, -0.2959, 0.0063, 0.2771, -0.0123, -0.0567, 0.2402, -0.0227, -0.1262, 0.1857, 0.1738, 0.2479, 0.0220, 0.1466, 0.1106, 0.2829, -0.2780, -0.0592, -0.2866, 0.2533, -0.1570, -0.0581, -0.1937, -0.2310, -0.2228, 0.2691, -0.1706, -0.2956, 0.1968, -0.0026, 0.1122, 0.0306, 0.2472, -0.0574, 0.1175, 0.2154, 0.0036, 0.0545, 0.0457, -0.2085, 0.0851, 0.0479, 0.0688, -0.1397, -0.1684, 0.2644, -0.1193, -0.0537, 0.0763, -0.2831, 0.1133, 0.1597, -0.1549, 0.0056, -0.1912, 0.1605, -0.0745, -0.2234, -0.2433, 0.1004, -0.1969, 0.0752, -0.0426, 0.0655, 0.2400, -0.0041, 0.2300, 0.1616, 0.0693, 0.2768, 0.0727, -0.2282, 0.0285, -0.2905, -0.1975, -0.0663, 0.1384, 0.0934, -0.2703, 0.2999, -0.1989, -0.2465, -0.1591, -0.1965, 0.2667, 0.1803, 0.2934, 0.1779, 0.2429, -0.2852, -0.2420, -0.2177, -0.1843, 0.2528, 0.1613, -0.0391, -0.2394, -0.1719, 0.0765, -0.0098, -0.2244, -0.0654, 0.2900, 0.2227, -0.0944, -0.0476, -0.2421, -0.0996, 0.2677, -0.2638, -0.2878, -0.1720, -0.2799, 0.0000, 0.0950, 0.0824, 0.1998, 0.2201, -0.0824, -0.0969, 0.0248, -0.2047, -0.1011, -0.0761, 0.0394, -0.1805, -0.0255, -0.2373, -0.0792, 0.0446, 0.2471, 0.1849, 0.2748, 0.0110, 0.2306, 0.2255, 0.2524, 0.2137, -0.1172, 0.1050, 0.0124, -0.1753, -0.0209, 0.0346, -0.0328, 0.2742, -0.2062, -0.1097, 0.2469, -0.2210, -0.2285, 0.1157, -0.2595, 0.2171, 0.0020, 0.2248, -0.2991, 0.0522, 0.1955, 0.1114, -0.1660, -0.1996, -0.2274, 0.2361, 0.1213, -0.0260, -0.2864, -0.2135, -0.1626, -0.3054, 0.1355, 0.1789, 0.0044, -0.1374, 0.0094, 0.1196, -0.0171, -0.2391, -0.0113, 0.1016, -0.0064, 0.1738, 0.2694, 0.2214, -0.2007, 0.2880, 0.1311, 0.1536, 0.0808, -0.0267, 0.2312, -0.2656, 0.1569, 0.1842, -0.0540, 0.3007, -0.1049, 0.0445, -0.2446, 0.1892, 0.1755, -0.1642, -0.0183, 0.1335, 0.3003, 0.1893, -0.3065, -0.1626, -0.0337, -0.0143, -0.0155, -0.1480, -0.1724, -0.1862, -0.1498, 0.2450, -0.2354, 0.0147, 0.1099, 0.0441, 0.3084, 0.0160, 0.0710, 0.1524, -0.2168, 0.1923, -0.1731, 0.0442, 0.2241, 0.1453, 0.2972, 0.0820, -0.1823, 0.1690, -0.2450, -0.0852, -0.1378, 0.2424, 0.2798, 0.0048, -0.2529, 0.1118, -0.2287, -0.1543, 0.0778, 0.2095, -0.0726, -0.2903, -0.0797, -0.2385, 0.0552, 0.2608, -0.1162, 0.0933, 0.2807, 0.2754, -0.2952, -0.0337, -0.2345, -0.0509, 0.1790, -0.0700, 0.2434, 0.1979, -0.2025, 0.0305, 0.1428, 0.1422, 0.1772, 0.0378, -0.0249, -0.0656, 0.0831, -0.2269, -0.2874, 0.0126, 0.2296, 0.1334, -0.1777, -0.2366, 0.1559, 0.3782, -0.2540, -0.2792, -0.0612, -0.2777, -0.3352, 0.1094, 0.0877, -0.0487, 0.3688, 0.2684, 0.1746, 0.0735, 0.3887, 0.3896, -0.1788, 0.0728, 0.1093, -0.1047, 0.2729, -0.2646, 0.3387, 0.2368, -0.3754, -0.0778, -0.3509, -0.1920, -0.3398};
    float pg[] = {0.000470, -0.001377, 0.003245, -0.003011, -0.000605, 0.000418, 0.000038, -0.000236, -0.000160, -0.000421, -0.000066, -0.000001, -0.000864, -0.000637, -0.000341, -0.000521, -0.000396, -0.000769, 0.000023, -0.000036, -0.000026, -0.000057, -0.000013, -0.000003, 0.000059, 0.000110, -0.000040, 0.000076, 0.000070, 0.000052, -0.000012, 0.000002, 0.000003, 0.000000, 0.000003, 0.000002, -0.001038, -0.000647, -0.000551, -0.000508, -0.000406, -0.000924, 0.000014, -0.000037, -0.000026, -0.000064, -0.000012, -0.000002, -0.011523, -0.010467, -0.000388, -0.011742, -0.006098, -0.010256, 0.000271, -0.000884, -0.000611, -0.001530, -0.000268, -0.000028, -0.001351, -0.001224, -0.000033, -0.001399, -0.000710, -0.001202, 0.000029, -0.000106, -0.000073, -0.000185, -0.000032, -0.000003, -0.001010, -0.000873, -0.000098, -0.000969, -0.000510, -0.000899, 0.000020, -0.000074, -0.000051, -0.000128, -0.000022, -0.000002, -0.000706, -0.001035, 0.000714, -0.001533, -0.000554, -0.000628, 0.000022, -0.000119, -0.000081, -0.000212, -0.000034, -0.000001, -0.009837, -0.010282, 0.001347, -0.011477, -0.005988, -0.008755, 0.000382, -0.000827, -0.000585, -0.001387, -0.000271, -0.000050, -0.001038, -0.001078, 0.000242, -0.001375, -0.000606, -0.000924, 0.000022, -0.000106, -0.000073, -0.000188, -0.000031, -0.000001, -0.000968, -0.000884, -0.000057, -0.000944, -0.000521, -0.000861, 0.000028, -0.000069, -0.000048, -0.000117, -0.000022, -0.000003, -0.001170, -0.001827, 0.001256, -0.002556, -0.000996, -0.001041, 0.000060, -0.000192, -0.000133, -0.000332, -0.000058, -0.000006, -0.006425, -0.003058, -0.003157, -0.004371, -0.001686, -0.005719, -0.000243, -0.000436, -0.000263, -0.000886, -0.000073, 0.000057, -0.003610, -0.002433, -0.001094, -0.002895, -0.001403, -0.003213, -0.000023, -0.000246, -0.000160, -0.000460, -0.000059, 0.000011, -0.001756, -0.000744, -0.000763, -0.001432, -0.000365, -0.001563, -0.000110, -0.000153, -0.000089, -0.000318, -0.000022, 0.000025, -0.004384, -0.002511, -0.001205, -0.004100, -0.001314, -0.003902, -0.000184, -0.000396, -0.000242, -0.000790, -0.000072, 0.000045, 0.000638, 0.002189, -0.002003, 0.002417, 0.001268, 0.000568, -0.000193, 0.000139, 0.000112, 0.000189, 0.000066, 0.000033, -0.000296, -0.000159, -0.000157, -0.000168, -0.000095, -0.000264, -0.000004, -0.000015, -0.000010, -0.000029, -0.000003, 0.000001, 0.000139, 0.000143, -0.000004, 0.000140, 0.000085, 0.000123, -0.000007, 0.000009, 0.000007, 0.000015, 0.000003, 0.000001, -0.000347, -0.000018, -0.000431, 0.000044, -0.000021, -0.000309, -0.000018, -0.000002, 0.000001, -0.000010, 0.000002, 0.000003, 0.004589, 0.004766, -0.001954, 0.007467, 0.002508, 0.004084, 0.000038, 0.000628, 0.000410, 0.001167, 0.000154, -0.000024, 0.000012, -0.000210, 0.000147, -0.000025, -0.000147, 0.000010, 0.000045, 0.000012, 0.000003, 0.000037, -0.000004, -0.000009, -0.000016, 0.000023, -0.000029, -0.000003, 0.000017, -0.000014, -0.000007, -0.000002, -0.000001, -0.000006, 0.000000, 0.000001, 0.000016, -0.000085, 0.000037, 0.000046, -0.000067, 0.000014, 0.000024, 0.000011, 0.000005, 0.000029, -0.000001, -0.000005, -0.089950, -0.009937, -0.008565, -0.006125, -0.020863, -0.001873, -0.000887, 0.007421, 0.007888, 0.004022, 0.001085, 0.009919, 0.003200, -0.001310, 0.011868, 0.001352, 0.001137, 0.000806, 0.002777, 0.000241, 0.000105, -0.000990, -0.001053, -0.000535, -0.000143, -0.001320, -0.000424, 0.000174, 0.008343, 0.000662, 0.000751, 0.000562, 0.001760, 0.000200, 0.000125, -0.000590, -0.000625, -0.000326, -0.000095, -0.000801, -0.000266, 0.000107, 0.011401, 0.001517, 0.001131, 0.000900, 0.002952, 0.000308, 0.000302, -0.001235, -0.001303, -0.000691, -0.000212, -0.001695, -0.000577, 0.000228, 0.055277, 0.004051, 0.004893, 0.002378, 0.009901, 0.000254, -0.001763, -0.001530, -0.001756, -0.000566, 0.000162, -0.001508, -0.000115, 0.000151, -0.004552, -0.000290, -0.000395, -0.000193, -0.000784, -0.000024, 0.000142, 0.000106, 0.000123, 0.000036, -0.000015, 0.000099, 0.000002, -0.000009, -0.001497, -0.000253, -0.000156, -0.000045, -0.000338, 0.000026, 0.000116, 0.000057, 0.000068, 0.000015, -0.000014, 0.000045, -0.000007, -0.000003, -0.005188, -0.000827, -0.000534, -0.000191, -0.001182, 0.000057, 0.000323, 0.000240, 0.000277, 0.000086, -0.000030, 0.000231, 0.000012, -0.000022, 0.000446, 0.003733, 0.000671, 0.000415, 0.002927, -0.000117, -0.000009, -0.001929, -0.002056, -0.001034, -0.000265, -0.002556, -0.000808, 0.000335, -0.000202, -0.000189, -0.000048, -0.000030, -0.000173, 0.000003, 0.000002, 0.000100, 0.000106, 0.000053, 0.000013, 0.000131, 0.000041, -0.000017, -0.000998, -0.000078, -0.000090, -0.000068, -0.000211, -0.000024, -0.000016, 0.000071, 0.000075, 0.000039, 0.000012, 0.000096, 0.000032, -0.000013, -0.000361, -0.000286, -0.000077, -0.000066, -0.000291, -0.000012, -0.000034, 0.000183, 0.000193, 0.000102, 0.000031, 0.000250, 0.000085, -0.000034, 0.043140, -0.003702, 0.002673, 0.002606, 0.004149, 0.001646, 0.001551, -0.000141, -0.000074, -0.000230, -0.000245, -0.000502, -0.000379, 0.000094, -0.000038, 0.000058, 0.000007, -0.000014, 0.000018, -0.000018, -0.000037, 0.000002, 0.000000, 0.000006, 0.000007, 0.000012, 0.000010, -0.000002, -0.000239, -0.000009, -0.000020, -0.000020, -0.000049, -0.000010, -0.000014, 0.000020, 0.000021, 0.000012, 0.000005, 0.000030, 0.000012, -0.000004, -0.000024, 0.000092, 0.000013, -0.000023, 0.000032, -0.000030, -0.000062, 0.000006, 0.000002, 0.000011, 0.000012, 0.000023, 0.000018, -0.000004, -0.003808, 0.012274, 0.001783, -0.000086, 0.007524, -0.001468, -0.002366, -0.004263, -0.004682, -0.002005, -0.000177, -0.005077, -0.001207, 0.000615, -0.003662, -0.002375, -0.000681, -0.000205, -0.002072, 0.000198, 0.000467, 0.000892, 0.000980, 0.000420, 0.000037, 0.001062, 0.000253, -0.000129, -0.005872, -0.000692, -0.000565, -0.000301, -0.001277, -0.000036, 0.000146, 0.000333, 0.000366, 0.000157, 0.000014, 0.000396, 0.000094, -0.000048, -0.003266, -0.003041, -0.000763, -0.000190, -0.002454, 0.000282, 0.000595, 0.001120, 0.001230, 0.000527, 0.000047, 0.001334, 0.000317, -0.000162, 0.004255, 0.010390, 0.002083, 0.000486, 0.007627, -0.000946, -0.001656, -0.004032, -0.004395, -0.001961, -0.000262, -0.004932, -0.001274, 0.000611, -0.002592, -0.000602, -0.000299, -0.000144, -0.000768, 0.000011, 0.000104, 0.000266, 0.000291, 0.000128, 0.000016, 0.000323, 0.000082, -0.000040, -0.001329, -0.000137, -0.000125, -0.000076, -0.000286, -0.000017, 0.000013, 0.000082, 0.000089, 0.000042, 0.000008, 0.000104, 0.000029, -0.000013, -0.003140, -0.000896, -0.000391, -0.000186, -0.001052, 0.000024, 0.000139, 0.000398, 0.000433, 0.000194, 0.000027, 0.000489, 0.000128, -0.000061, -0.036476, 0.008151, -0.001415, -0.002363, -0.000445, -0.002129, -0.002683, -0.001307, -0.001538, -0.000405, 0.000252, -0.001129, 0.000064, 0.000094, 0.000012, 0.000183, 0.000032, -0.000002, 0.000117, -0.000024, -0.000044, -0.000057, -0.000063, -0.000025, -0.000000, -0.000065, -0.000013, 0.000008, 0.000090, 0.000031, 0.000012, -0.000004, 0.000023, -0.000009, -0.000023, 0.000002, 0.000001, 0.000004, 0.000005, 0.000009, 0.000007, -0.000002, 0.000023, 0.000226, 0.000040, -0.000008, 0.000140, -0.000034, -0.000065, -0.000062, -0.000070, -0.000025, 0.000003, -0.000066, -0.000010, 0.000007, 0.778076, -0.049629, -0.055745, -0.026004, -0.005415, -0.064495, -0.016446, 0.005688, 1.031509, -0.093285, -0.099193, -0.058751, -0.024756, -0.141341, -0.052230, 0.015920, -0.034874, 0.004444, 0.006071, 0.003392, 0.001704, 0.007931, 0.001726, 0.000686, 0.835289, -0.053310, -0.059046, -0.027800, -0.005810, -0.069013, -0.018484, 0.006991};  
    float rg_t1[] = {0.1676, -0.0309, -0.1636, 0.2121, -0.4894, -0.3723, -0.3160};
    float rg_t2[] = {-0.1355, 0.2009, 0.0175, -0.0996, 0.3757, 0.2478, 0.2300};
    float rg_t3[] = {-0.2468, -0.0570, 0.0505, 0.0640, -0.0004, 0.0274, -0.0749};
    float cg_t1[] = {3.1341, -1.2491, -0.1288, 4.1868};
    float cg_t2[] = {-1.0711, 4.4837, 1.1587, -0.5370};
    float cg_t3[] = {1.5070, 3.3692, -1.2134, 1.4436};

    //float ig_l1_t1 = 
    //float ig_l1_t1 = 
    //float ig_l1_t1 = 

    LSTM n = create_lstm(5, 6, 7, 4);
    n.seq_len = 3;
    n.stateful = 1;
    assign_array(p, n.params, n.num_params);

    wipe(&n);
    printf("Testing forward pass functionality...\n\n");
    sleep(1);
    lstm_forward(&n, x1);

    float *tmp = retrieve_array(n.layers[0].output[0], n.layers[0].size);
    if(!assert_equal(tmp, l1_t1, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden LSTM layer output was incorrect on 0th timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: First hidden LSTM layer output was as expected on 0th timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output[0], n.layers[1].size);
    if(!assert_equal(tmp, l2_t1, n.layers[1].size)){
      printf("X | TEST FAILED: Second hidden LSTM layer output was incorrect on 0th timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: Second hidden LSTM layer output was as expected on 0th timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.output_layer.output, n.output_dimension);
    if(!assert_equal(tmp, l3_t1, n.output_dimension)){
      printf("X | TEST FAILED: LSTM output layer output was incorrect on 0th timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM output layer output was as expected on 0th timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    if(!assert_equal(n.output, l3_t1, n.output_dimension)){
      printf("X | TEST FAILED: LSTM network output was incorrect on 0th timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM network output was as expected on 0th timestep (without incrementing n.t).\n");
    }

    printf("\n");
    lstm_forward(&n, x2);

    tmp = retrieve_array(n.layers[0].output[0], n.layers[0].size);
    if(!assert_equal(tmp, l1_t2, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden LSTM layer output was incorrect on 1st timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: First hidden LSTM layer output was as expected on 1st timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output[0], n.layers[1].size);
    if(!assert_equal(tmp, l2_t2, n.layers[0].size)){
      printf("X | TEST FAILED: Second hidden LSTM layer output was incorrect on 1st timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: Second hidden LSTM layer output was as expected on 1st timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.output_layer.output, n.output_dimension);
    if(!assert_equal(tmp, l3_t2, n.output_dimension)){
      printf("X | TEST FAILED: LSTM output layer output was incorrect on 1st timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM output layer output was as expected on 1st timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    if(!assert_equal(n.output, l3_t2, n.output_dimension)){
      printf("X | TEST FAILED: LSTM network output was incorrect on 1st timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM network output was as expected on 1st timestep (without incrementing n.t).\n");
    }

    printf("\n");
    lstm_forward(&n, x3);

    tmp = retrieve_array(n.layers[0].output[0], n.layers[0].size);
    if(!assert_equal(tmp, l1_t3, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden LSTM layer output was incorrect on 2nd timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: First hidden LSTM layer output was as expected on 2nd timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output[0], n.layers[1].size);
    if(!assert_equal(tmp, l2_t3, n.layers[0].size)){
      printf("X | TEST FAILED: Second hidden LSTM layer output was incorrect on 2nd timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: Second hidden LSTM layer output was as expected on 2nd timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.output_layer.output, n.output_dimension);
    if(!assert_equal(tmp, l3_t3, n.output_dimension)){
      printf("X | TEST FAILED: LSTM output layer output was incorrect on 2nd timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM output layer output was as expected on 2nd timestep (without incrementing n.t).\n");
    }
    dispose_array(tmp);

    if(!assert_equal(n.output, l3_t3, n.output_dimension)){
      printf("X | TEST FAILED: LSTM network output was incorrect on 2nd timestep (without incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM network output was as expected on 2nd timestep (without incrementing n.t).\n");
    }

    sleep(1);
    printf("\ntesting forward + backward pass in conjunction...\n");
    sleep(1);
    wipe(&n);
    //LSTM n = create_lstm(5, 6, 7, 4);
    //assign_array(p, n.params, n.num_params);

    printf("\n");
    lstm_forward(&n, x1);
    float c1 = lstm_cost(&n, y1) - 3.361277;
    //float c0 = lstm_cost(&n, y1) - 3.361277;
    //printf("successuflly got cost %f\n", c0);

    tmp = retrieve_array(n.layers[0].output[0], n.layers[0].size);
    if(!assert_equal(tmp, l1_t1, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden LSTM layer output was incorrect on 0th timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: First hidden LSTM layer output was as expected on 0th timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output[0], n.layers[1].size);
    if(!assert_equal(tmp, l2_t1, n.layers[1].size)){
      printf("X | TEST FAILED: Second hidden LSTM layer output was incorrect on 0th timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: Second hidden LSTM layer output was as expected on 0th timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.output_layer.output, n.output_dimension);
    if(!assert_equal(tmp, l3_t1, n.output_dimension)){
      printf("X | TEST FAILED: LSTM output layer output was incorrect on 0th timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM output layer output was as expected on 0th timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    if(!assert_equal(n.output, l3_t1, n.output_dimension)){
      printf("X | TEST FAILED: LSTM network output was incorrect on 0th timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM network output was as expected on 0th timestep (while incrementing n.t).\n");
    }
    printf("\n");
    if(c1 < 0) c1*=-1;
    if(c1 > 0.0001){
      printf("X | TEST FAILED: Cost scalar was not calculated correctly for 0th timestep.\n");
    }else{
      printf("  | TEST PASSED: Cost scalar was as expected for 0th timestep.\n");
    }

    tmp = retrieve_array(n.cost_gradient, n.output_dimension);
    if(!assert_equal(tmp, cg_t1, n.output_dimension)){
      printf("X | TEST FAILED: LSTM cost gradient was incorrect on 0th timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM cost gradient was as expected on 0th timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.recurrent_gradient[0], n.layers[n.depth-1].size);
    if(!assert_equal(tmp, rg_t1, n.layers[n.depth-1].size)){
      printf("X | TEST FAILED: LSTM recurrent layer gradient was incorrect on 0th timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM recurrent layer gradient was as expected on 0th timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);
    printf("\n");

    lstm_forward(&n, x2);

    tmp = retrieve_array(n.layers[0].output[1], n.layers[0].size);
    if(!assert_equal(tmp, l1_t2, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden LSTM layer output was incorrect on 1st timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: First hidden LSTM layer output was as expected on 1st timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output[1], n.layers[1].size);
    if(!assert_equal(tmp, l2_t2, n.layers[0].size)){
      printf("X | TEST FAILED: Second hidden LSTM layer output was incorrect on 1st timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: Second hidden LSTM layer output was as expected on 1st timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.output_layer.output, n.output_dimension);
    if(!assert_equal(tmp, l3_t2, n.output_dimension)){
      printf("X | TEST FAILED: LSTM output layer output was incorrect on 1st timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM output layer output was as expected on 1st timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    if(!assert_equal(n.output, l3_t2, n.output_dimension)){
      printf("X | TEST FAILED: LSTM network output was incorrect on 1st timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM network output was as expected on 1st timestep (while incrementing n.t).\n");
    }

    printf("\n");
    float c2 = lstm_cost(&n, y2) - 3.155137;
    if(c2 < 0) c2*=-1;
    if(c2 > 0.0001){
      printf("X | TEST FAILED: Cost scalar was not calculated correctly for 1st timestep.\n");
    }else{
      printf("  | TEST PASSED: Cost scalar was as expected for 1st timestep.\n");
    }

    tmp = retrieve_array(n.cost_gradient, n.output_dimension);
    if(!assert_equal(tmp, cg_t2, n.output_dimension)){
      printf("X | TEST FAILED: LSTM cost gradient was incorrect on 1st timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM cost gradient was as expected on 1st timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.recurrent_gradient[1], n.layers[n.depth-1].size);
    if(!assert_equal(tmp, rg_t2, n.layers[n.depth-1].size)){
      printf("X | TEST FAILED: LSTM recurrent layer gradient was incorrect on 1st timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM recurrent layer gradient was as expected on 1st timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);
    printf("\n");

    lstm_forward(&n, x3);

    tmp = retrieve_array(n.layers[0].output[2], n.layers[0].size);
    if(!assert_equal(tmp, l1_t3, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden LSTM layer output was incorrect on 2nd timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: First hidden LSTM layer output was as expected on 2nd timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output[2], n.layers[1].size);
    if(!assert_equal(tmp, l2_t3, n.layers[0].size)){
      printf("X | TEST FAILED: Second hidden LSTM layer output was incorrect on 2nd timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: Second hidden LSTM layer output was as expected on 2nd timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.output_layer.output, n.output_dimension);
    if(!assert_equal(tmp, l3_t3, n.output_dimension)){
      printf("X | TEST FAILED: LSTM output layer output was incorrect on 2nd timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM output layer output was as expected on 2nd timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    if(!assert_equal(n.output, l3_t3, n.output_dimension)){
      printf("X | TEST FAILED: LSTM network output was incorrect on 2nd timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM network output was as expected on 2nd timestep (while incrementing n.t).\n");
    }

    printf("\n");
    float c3 = lstm_cost(&n, y3) - 3.286844;
    if(c3 < 0) c3*=-1;
    if(c3 > 0.0001){
      printf("X | TEST FAILED: Cost scalar was not calculated correctly for 2nd timestep.\n");
    }else{
      printf("  | TEST PASSED: Cost scalar was as expected for 2nd timestep.\n");
    }

    tmp = retrieve_array(n.cost_gradient, n.output_dimension);
    if(!assert_equal(tmp, cg_t3, n.output_dimension)){
      printf("X | TEST FAILED: LSTM cost gradient was incorrect on 2nd timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM cost gradient was as expected on 2nd timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.recurrent_gradient[2], n.layers[n.depth-1].size);
    if(!assert_equal(tmp, rg_t3, n.layers[n.depth-1].size)){
      printf("X | TEST FAILED: LSTM recurrent layer gradient was incorrect on 2nd timestep (while incrementing n.t).\n");
    }else{
      printf("  | TEST PASSED: LSTM recurrent layer gradient was as expected on 2nd timestep (while incrementing n.t).\n");
    }
    dispose_array(tmp);
    printf("\n");

    float gl1t1[] = {0.000989, -0.001302, -0.000484, 0.001409, 0.001318, -0.000321, 0.001905, -0.002441, 0.001614, 0.001433, -0.000243};
    float gl1t2[] = {-0.002286, 0.000806, 0.000763, -0.003137, -0.002644, -0.001046, -0.003120, 0.000631, 0.000080, -0.002541, -0.000950};
    float gl1t3[] = {-0.001039, 0.001392, 0.000891, -0.002091, -0.001781, -0.000337, 0.000035, 0.000005, 0.000543, 0.000330, 0.000369};
    float gl2t1[] = {0.023222, 0.029418, 0.047154, 0.007058, -0.020300, -0.026618, -0.032539, -0.006546, -0.034343, 0.000351, 0.030822, 0.015958, 0.044141};
    float gl2t2[] = {-0.008424, -0.037569, -0.071121, -0.020271, 0.028489, 0.010587, 0.027644, 0.023600, 0.017069, 0.004688, -0.040968, -0.033726, -0.025105};
    float gl2t3[] = {-0.002151, -0.007519, -0.007009, -0.022567, -0.004215, 0.009594, -0.010958, 0.001007, 0.016882, -0.011718, -0.008547, 0.002823, -0.000037};
    lstm_backward(&n);

    tmp = retrieve_array(n.layers[0].input_gradient[0], n.layers[0].input_dimension);
    if(!assert_equal(tmp, gl1t1, n.layers[0].input_dimension)){
      printf("X | TEST FAILED: Layer 1 input gradient at t 0 was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Layer 1 input gradient at t 0 was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[0].input_gradient[1], n.layers[0].input_dimension);
    if(!assert_equal(tmp, gl1t2, n.layers[0].input_dimension)){
      printf("X | TEST FAILED: Layer 1 input gradient at t 1 was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Layer 1 input gradient at t 1 was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[0].input_gradient[2], n.layers[0].input_dimension);
    if(!assert_equal(tmp, gl1t3, n.layers[0].input_dimension)){
      printf("X | TEST FAILED: Layer 1 input gradient at t 2 was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Layer 1 input gradient at t 2 was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].input_gradient[0], n.layers[1].input_dimension);
    if(!assert_equal(tmp, gl2t1, n.layers[1].input_dimension)){
      printf("X | TEST FAILED: Layer 1 input gradient at t 0 was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Layer 1 input gradient at t 0 was as expected.\n");
    }
    dispose_array(tmp);
    tmp = retrieve_array(n.layers[1].input_gradient[1], n.layers[1].input_dimension);
    if(!assert_equal(tmp, gl2t2, n.layers[1].input_dimension)){
      printf("X | TEST FAILED: Layer 1 input gradient at t 1 was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Layer 1 input gradient at t 1 was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].input_gradient[2], n.layers[1].input_dimension);
    if(!assert_equal(tmp, gl2t3, n.layers[1].input_dimension)){
      printf("X | TEST FAILED: Layer 1 input gradient at t 2 was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Layer 1 input gradient at t 2 was as expected.\n");
    }
    dispose_array(tmp);


    printf("\n");

    tmp = retrieve_array(n.param_grad, n.num_params);
    //PRINTLIST(tmp, n.num_params);
    //PRINTLIST(pg, n.num_params);
    if(!assert_equal(tmp, pg, n.num_params)){
      printf("X | TEST FAILED: LSTM parameter gradient was incorrect.\n");
    }else{
      printf("  | TEST PASSED: LSTM parameter gradient was as expected.\n");
    }
    dispose_array(tmp);

    save_lstm(&n, "./model/test.lstm");

    LSTM m = load_lstm("./model/test.lstm");

    float *tmp1 = retrieve_array(n.params, n.num_params);
    float *tmp2 = retrieve_array(m.params, m.num_params);
    if(!assert_equal(tmp1, tmp2, n.num_params)){
      printf("X | TEST FAILED: LSTM saved to disk did not match once loaded back into memory.\n");
    }else{
      printf("  | TEST PASSED: LSTM saved to disk was identical to original.\n");
    }
    dispose_array(tmp1);
    dispose_array(tmp2);
    sleep(1);

    dealloc_lstm(&n);
    dealloc_lstm(&m);
  }
}
