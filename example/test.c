/* Jonah Siekmann
 * 1/20/2019
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <lstm.h>
#include <optimizer.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;
#define PRINTLIST(name, len) printf("printing %s: {", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("};\n");}

float uniform(float minimum, float maximum){
	float center = minimum + (maximum - minimum)/2;
	float max_mag = maximum - center;
	if(rand()&1)
		return center + ((((float)rand())/RAND_MAX)) * max_mag;
	else
		return center - ((((float)rand())/RAND_MAX)) * max_mag;
}

int assert_equal(float *v1, float *v2, size_t len){
	for(int i = 0; i < len; i++){
		float diff = v1[i] - v2[i];
		if(diff < 0) diff*=-1;
		if(diff > 0.0001){
			printf("%f is not %f\n", v1[i], v2[i]);
			return 0;
		}
	}
	return 1;
}

#ifndef GPU

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
	return tmp;
}
void assign_array(float *arr, cl_mem dest, size_t len){
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), dest, 1, 0, sizeof(float) * len, arr, 0, NULL, NULL), "enqueuing cost gradient");
}
void dispose_array(float *arr){
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
	printf("preparing to run unit test suite...\n");
	sleep(1);

	srand(1);
	/* MLP tests */
	{
		printf("\n******** TESTING MLP FUNCTIONALITY ********\n\n");
		sleep(1);
		MLP n = create_mlp(4, 6, 7, 5);
		
		int success = 1;
		for(int i = 0; i < n.depth; i++){
			if(i < n.depth-1 && n.layers[i].logistic != sigmoid){
				printf("X | TEST FAILED: Default hidden layer logistic function was not sigmoid!\n");
				success = 0;
			}else if(i == n.depth-1 && n.layers[i].logistic != softmax){
				printf("X | TEST FAILED: Default output layer logistic function was not softmax!\n");
				success = 0;
			}
		}
		if(success)
			printf("  | TEST PASSED: Default logistic functions were as expected.\n");

		float x[]  = {0.50, 1.2, 0.01, 0.75};
		float y[]  = {0.25, 0.09, 0.0, 1.0, 0.75};
		float l0[] = {0.4103, 0.5631, 0.4348, 0.5297, 0.3943, 0.4167};
		float l1[] = {0.3545, 0.5293, 0.5700, 0.6833, 0.3192, 0.4952, 0.4268};
		float l2[] = {0.3459, 0.1410, 0.0896, 0.2343, 0.1892};
		float c[]  = {-0.4238, -0.4210, -1.0984, 4.2683, 3.6548};
		float g0[] = {0.0020, 0.0047, 0.0011, 0.0011};
		float g1[] = {-0.0319, 0.0119, 0.0122, 0.0136, 0.0118, -0.0174};
		float g2[] = {0.3366, 0.0155, -0.0865, 0.4037, -0.0830, -0.1082, -0.0088};
		float p[]  = {0.3583, -0.3339, -0.3887, 0.1429, -0.1184, -0.2036, -0.1555, 0.4060, 0.2711, 0.0604, 0.0070, -0.0585, 0.0668, -0.0553, -0.4259, 0.2187, 0.2612, -0.2719, 0.2105, 0.1247, -0.2246, -0.1707, -0.1208, 0.3444, 0.0297, 0.2243, 0.0820, -0.3796, 0.0274, -0.1952, -0.0901, -0.3410, -0.1008, -0.1418, -0.1937, -0.2009, -0.1654, -0.3518, 0.1075, 0.2419, 0.2600, 0.1663, 0.3134, -0.0865, -0.1324, 0.3615, 0.2484, 0.1661, 0.1506, -0.2586, 0.1824, 0.3592, -0.0558, 0.2423, 0.2342, 0.2971, 0.1690, -0.0709, -0.2103, -0.0641, -0.0390, -0.1873, -0.3722, -0.2587, -0.2834, 0.1112, -0.2209, -0.0576, -0.0474, 0.0620, -0.0282, -0.0199, -0.0666, 0.3015, -0.2482, -0.2417, -0.0353, -0.1966, -0.0264, -0.1810, 0.2249, 0.0204, 0.3922, 0.3491, 0.3914, 0.3415, 0.0016, -0.2329, -0.0640, 0.3581, -0.1408, 0.2273, -0.2696, -0.2082, -0.1194, -0.2263, -0.2933, 0.0139, -0.3268, -0.3425, 0.3842, -0.3543, 0.2616, 0.0643, 0.3487, 0.2548, -0.2469, 0.2748, -0.1290, -0.0291, -0.0877, -0.2003, 0.1098, -0.2821, 0.1849, 0.3704, 0.1319, -0.1704, 0.1353};
		float pg[] = {-0.0077, -0.0039, -0.0093, -0.0001, -0.0058, 0.0029, 0.0015, 0.0035, 0.0000, 0.0022, 0.0030, 0.0015, 0.0036, 0.0000, 0.0022, 0.0034, 0.0017, 0.0041, 0.0000, 0.0025, 0.0028, 0.0014, 0.0034, 0.0000, 0.0021, -0.0042, -0.0021, -0.0051, -0.0000, -0.0032, 0.0770, 0.0316, 0.0434, 0.0335, 0.0408, 0.0304, 0.0321, 0.0039, 0.0016, 0.0022, 0.0017, 0.0020, 0.0015, 0.0016, -0.0212, -0.0087, -0.0119, -0.0092, -0.0112, -0.0084, -0.0088, 0.0874, 0.0359, 0.0492, 0.0380, 0.0463, 0.0345, 0.0364, -0.0180, -0.0074, -0.0102, -0.0078, -0.0096, -0.0071, -0.0075, -0.0271, -0.0111, -0.0152, -0.0118, -0.0143, -0.0107, -0.0113, -0.0021, -0.0009, -0.0012, -0.0009, -0.0011, -0.0008, -0.0009, -0.0959, -0.0340, -0.0508, -0.0547, -0.0655, -0.0306, -0.0475, -0.0409, -0.0510, -0.0181, -0.0270, -0.0291, -0.0348, -0.0163, -0.0252, -0.0218, -0.0896, -0.0318, -0.0474, -0.0511, -0.0612, -0.0286, -0.0444, -0.0382, 0.7657, 0.2715, 0.4053, 0.4365, 0.5232, 0.2445, 0.3792, 0.3268, 0.5608, 0.1988, 0.2968, 0.3196, 0.3832, 0.1790, 0.2777, 0.2393};

		assign_array(p, n.params, n.num_params);
		
		mlp_forward(&n, x);
		float *tmp = retrieve_array(n.layers[0].output, n.layers[0].size);
		if(!assert_equal(tmp, l0, n.layers[0].size)){
			printf("X | TEST FAILED: First hidden layer's output was incorrect.\n");
		}else{
			printf("  | TEST PASSED: First hidden layer's output was as expected.\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.layers[1].output, n.layers[1].size);
		if(!assert_equal(tmp, l1, n.layers[1].size)){
			printf("X | TEST FAILED: Second hidden layer's output was incorrect.\n");
		}else{
			printf("  | TEST PASSED: Second hidden layer's output was as expected.\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.layers[2].output, n.layers[2].size);
		if(!assert_equal(tmp, l2, n.layers[2].size)){
			printf("X | TEST FAILED: Third hidden layer's output was incorrect.\n");
		}else{
			printf("  | TEST PASSED: Third hidden layer's output was as expected.\n");
		}
		dispose_array(tmp);

		printf("\n");
		mlp_cost(&n, y);

		if(!assert_equal(n.cost_gradient, c, n.output_dimension)){
			printf("X | TEST FAILED: Cost gradient was not calculated correcly.\n");
		}else{
			printf("  | TEST PASSED: Cost gradient was as expected.\n");
		}

		printf("\n");
		mlp_backward(&n);

		tmp = retrieve_array(n.layers[0].input_gradient, n.layers[0].input_dimension);
		if(!assert_equal(tmp, g0, n.layers[0].input_dimension)){
			printf("X | TEST FAILED: First hidden layer's input gradient was incorrect.\n");
		}else{
			printf("  | TEST PASSED: First hidden layer's input gradient was as expected.\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.layers[1].input_gradient, n.layers[1].input_dimension);
		if(!assert_equal(tmp, g1, n.layers[1].input_dimension)){
			printf("X | TEST FAILED: Second hidden layer's input gradient was incorrect.\n");
		}else{
			printf("  | TEST PASSED: Second hidden layer's input gradient was as expected.\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.layers[2].input_gradient, n.layers[2].input_dimension);
		if(!assert_equal(tmp, g2, n.layers[2].input_dimension)){
			printf("X | TEST FAILED: Third hidden layer's input gradient was incorrect.\n");
		}else{
			printf("  | TEST PASSED: Third hidden layer's input gradient was as expected.\n");
		}
		dispose_array(tmp);
		
		tmp = retrieve_array(n.param_grad, n.num_params);
		if(!assert_equal(tmp, pg, n.num_params)){
			printf("X | TEST FAILED: Network parameter gradients were incorrect.\n");
		}else{
			printf("  | TEST PASSED: Network parameter gradients were as expected.\n");
		}

		//TODO - tests for saving/loading file, once gpu dealloc works
		sleep(1);
	}
	/* LSTM tests */
	{
		printf("\n******** TESTING LSTM FUNCTIONALITY ********\n\n");
		sleep(1);
		LSTM n = create_lstm(5, 6, 7, 4);
		n.seq_len = 3;
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
		float pg[] = {0.0005, -0.0014, 0.0032, -0.0030, -0.0006, 0.0004, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0009, -0.0006, -0.0003, -0.0005, -0.0004, -0.0008, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0001, -0.0000, 0.0001, 0.0001, 0.0001, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0010, -0.0006, -0.0006, -0.0005, -0.0004, -0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0115, -0.0105, -0.0004, -0.0117, -0.0061, -0.0103, -0.0009, -0.0009, -0.0009, -0.0009, -0.0009, -0.0009, -0.0014, -0.0012, -0.0000, -0.0014, -0.0007, -0.0012, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0010, -0.0009, -0.0001, -0.0010, -0.0005, -0.0009, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0007, -0.0010, 0.0007, -0.0015, -0.0006, -0.0006, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0098, -0.0103, 0.0013, -0.0115, -0.0060, -0.0088, -0.0006, -0.0006, -0.0006, -0.0006, -0.0006, -0.0006, -0.0010, -0.0011, 0.0002, -0.0014, -0.0006, -0.0009, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0010, -0.0009, -0.0001, -0.0009, -0.0005, -0.0009, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0012, -0.0018, 0.0013, -0.0026, -0.0010, -0.0010, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0064, -0.0031, -0.0032, -0.0044, -0.0017, -0.0057, -0.0009, -0.0009, -0.0009, -0.0009, -0.0009, -0.0009, -0.0036, -0.0024, -0.0011, -0.0029, -0.0014, -0.0032, -0.0005, -0.0005, -0.0005, -0.0005, -0.0005, -0.0005, -0.0018, -0.0007, -0.0008, -0.0014, -0.0004, -0.0016, -0.0003, -0.0003, -0.0003, -0.0003, -0.0003, -0.0003, -0.0044, -0.0025, -0.0012, -0.0041, -0.0013, -0.0039, -0.0008, -0.0008, -0.0008, -0.0008, -0.0008, -0.0008, 0.0006, 0.0022, -0.0020, 0.0024, 0.0013, 0.0006, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0003, -0.0002, -0.0002, -0.0002, -0.0001, -0.0003, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, 0.0001, 0.0001, -0.0000, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0003, -0.0000, -0.0004, 0.0000, -0.0000, -0.0003, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0046, 0.0048, -0.0020, 0.0075, 0.0025, 0.0041, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, 0.0000, -0.0002, 0.0001, -0.0000, -0.0001, 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0001, 0.0000, 0.0000, -0.0001, 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0900, -0.0099, -0.0086, -0.0061, -0.0209, -0.0019, -0.0009, 0.0074, 0.0074, 0.0074, 0.0074, 0.0074, 0.0074, 0.0074, 0.0119, 0.0014, 0.0011, 0.0008, 0.0028, 0.0002, 0.0001, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, 0.0083, 0.0007, 0.0008, 0.0006, 0.0018, 0.0002, 0.0001, -0.0006, -0.0006, -0.0006, -0.0006, -0.0006, -0.0006, -0.0006, 0.0114, 0.0015, 0.0011, 0.0009, 0.0030, 0.0003, 0.0003, -0.0012, -0.0012, -0.0012, -0.0012, -0.0012, -0.0012, -0.0012, 0.0553, 0.0041, 0.0049, 0.0024, 0.0099, 0.0003, -0.0018, -0.0018, -0.0018, -0.0018, -0.0018, -0.0018, -0.0018, -0.0018, -0.0046, -0.0003, -0.0004, -0.0002, -0.0008, -0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0015, -0.0003, -0.0002, -0.0000, -0.0003, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0052, -0.0008, -0.0005, -0.0002, -0.0012, 0.0001, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0037, 0.0007, 0.0004, 0.0029, -0.0001, -0.0000, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, -0.0002, -0.0002, -0.0000, -0.0000, -0.0002, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0010, -0.0001, -0.0001, -0.0001, -0.0002, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0004, -0.0003, -0.0001, -0.0001, -0.0003, -0.0000, -0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0431, -0.0037, 0.0027, 0.0026, 0.0041, 0.0016, 0.0016, -0.0002, -0.0002, -0.0002, -0.0002, -0.0002, -0.0002, -0.0002, -0.0000, 0.0001, 0.0000, -0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0002, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0001, 0.0000, -0.0000, 0.0000, -0.0000, -0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0038, 0.0123, 0.0018, -0.0001, 0.0075, -0.0015, -0.0024, -0.0051, -0.0051, -0.0051, -0.0051, -0.0051, -0.0051, -0.0051, -0.0037, -0.0024, -0.0007, -0.0002, -0.0021, 0.0002, 0.0005, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, -0.0059, -0.0007, -0.0006, -0.0003, -0.0013, -0.0000, 0.0001, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, -0.0033, -0.0030, -0.0008, -0.0002, -0.0025, 0.0003, 0.0006, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0043, 0.0104, 0.0021, 0.0005, 0.0076, -0.0009, -0.0017, -0.0013, -0.0013, -0.0013, -0.0013, -0.0013, -0.0013, -0.0013, -0.0026, -0.0006, -0.0003, -0.0001, -0.0008, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0013, -0.0001, -0.0001, -0.0001, -0.0003, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0031, -0.0009, -0.0004, -0.0002, -0.0011, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0365, 0.0082, -0.0014, -0.0024, -0.0004, -0.0021, -0.0027, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0000, 0.0002, 0.0000, -0.0000, 0.0001, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0002, 0.0000, -0.0000, 0.0001, -0.0000, -0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7781, -0.0496, -0.0557, -0.0260, -0.0054, -0.0645, -0.0164, 0.0057, 1.0315, -0.0933, -0.0992, -0.0588, -0.0248, -0.1413, -0.0522, 0.0159, -0.0349, 0.0044, 0.0061, 0.0034, 0.0017, 0.0079, 0.0017, 0.0007, 0.8353, -0.0533, -0.0590, -0.0278, -0.0058, -0.0690, -0.0185, 0.0070};
		float rg_t1[] = {0.1676, -0.0309, -0.1636, 0.2121, -0.4894, -0.3723, -0.3160};
		float rg_t2[] = {-0.1355, 0.2009, 0.0175, -0.0996, 0.3757, 0.2478, 0.2300};
		float rg_t3[] = {-0.2468, -0.0570, 0.0505, 0.0640, -0.0004, 0.0274, -0.0749};
		float cg_t1[] = {3.1341, -1.2491, -0.1288, 4.1868};
		float cg_t2[] = {-1.0711, 4.4837, 1.1587, -0.5370};
		float cg_t3[] = {1.5070, 3.3692, -1.2134, 1.4436};

		//float ig_l1_t1 = 
		//float ig_l1_t1 = 
		//float ig_l1_t1 = 

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

		printf("\n");
		lstm_forward(&n, x1);

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
		dispose_array(tmp);

		printf("\n");
		lstm_cost(&n, y1);

		tmp = retrieve_array(n.mlp_cost_gradient, n.output_dimension);
		if(!assert_equal(tmp, cg_t1, n.output_dimension)){
			printf("X | TEST FAILED: LSTM cost gradient was incorrect on 0th timestep (while incrementing n.t).\n");
		}else{
			printf("  | TEST PASSED: LSTM cost gradient was as expected on 0th timestep (while incrementing n.t).\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.network_gradient[0], n.layers[n.depth-1].size);
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
		lstm_cost(&n, y2);

		tmp = retrieve_array(n.mlp_cost_gradient, n.output_dimension);
		if(!assert_equal(tmp, cg_t2, n.output_dimension)){
			printf("X | TEST FAILED: LSTM cost gradient was incorrect on 1st timestep (while incrementing n.t).\n");
		}else{
			printf("  | TEST PASSED: LSTM cost gradient was as expected on 1st timestep (while incrementing n.t).\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.network_gradient[1], n.layers[n.depth-1].size);
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
		lstm_cost(&n, y3);

		tmp = retrieve_array(n.mlp_cost_gradient, n.output_dimension);
		if(!assert_equal(tmp, cg_t3, n.output_dimension)){
			printf("X | TEST FAILED: LSTM cost gradient was incorrect on 2nd timestep (while incrementing n.t).\n");
		}else{
			printf("  | TEST PASSED: LSTM cost gradient was as expected on 2nd timestep (while incrementing n.t).\n");
		}
		dispose_array(tmp);

		tmp = retrieve_array(n.network_gradient[2], n.layers[n.depth-1].size);
		if(!assert_equal(tmp, rg_t3, n.layers[n.depth-1].size)){
			printf("X | TEST FAILED: LSTM recurrent layer gradient was incorrect on 2nd timestep (while incrementing n.t).\n");
		}else{
			printf("  | TEST PASSED: LSTM recurrent layer gradient was as expected on 2nd timestep (while incrementing n.t).\n");
		}
		dispose_array(tmp);
		printf("\n");
		lstm_backward(&n);

		tmp = retrieve_array(n.param_grad, n.num_params);
		if(!assert_equal(tmp, pg, n.num_params)){
			printf("X | TEST FAILED: LSTM parameter gradient was incorrect.\n");
		}else{
			printf("  | TEST PASSED: LSTM parameter gradient was as expected.\n");
		}
		dispose_array(tmp);
		sleep(1);
	}


}
