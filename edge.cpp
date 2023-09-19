/*

Licensed under a Creative Commons Attribution-ShareAlike 4.0
International License.

Code by James Reinders, for class at Cornell in September
2023. Based on Exercise 15 of SYCL Academy Code Exercises.

*/

/*******************************************************************

Check https://tinyurl.com/reinders-4class for lots of
information, only some of it is useful for this class.  :)




Known issues:

Crude addition of "blurred_" to front of file name won't
work if there is a directory in the path, so such runs are
rejected.

If the image is too large - the runtime may segment fault -
this code doesn't check for limits (bad, bad, bad!)

********************************************************************/
 
//#define MYDEBUGS
//#define DEBUGDUMP
//#define DOUBLETROUBLE
#define CORRECTNESS
#define PROFILE

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>

#include "image_conv.h"

inline constexpr int filterWidth = 16;
inline constexpr int halo = filterWidth / 2;

int main(int argc, char* argv[]) {
  const char* inFile = argv[1];
  char* outFile;
  int arg_ndevs;

  if (argc == 3) {
    if (strchr(inFile, '/') || strchr(inFile, '\\')) {
      std::cerr << "Sorry, filename cannot include a path.\n";
      exit(1);
    }
    const char* prefix = "blurred_";
    size_t len1 = strlen(inFile);
    size_t len2 = strlen(prefix);
    outFile = (char*)malloc((len1 + len2 + 1) * sizeof(char));
    strcpy(outFile, prefix);
    strcpy(outFile + 8, inFile);
    arg_ndevs = std::atoi(argv[2]);
#ifdef MYDEBUGS
    std::cout << "Input file: " << inFile << "\nOutput file: " << outFile
              << "\n";
#endif
  } else {
    std::cerr << "Usage: " << argv[0] << " imagefile\n";
    exit(1);
  }

  auto inImage = util::read_image(inFile, halo);

  auto outImage = util::allocate_image(inImage.width(), inImage.height(),
                                       inImage.channels());

  // The image convolution support code provides a
  // `filter_type` enum which allows us to choose between
  // `identity` and `blur`. The utility for generating the
  // filter data; `generate_filter` takes a `filter_type`
  // and a width.

  auto filter = util::generate_filter(util::filter_type::blur, filterWidth,
                                      inImage.channels());


  //
  // This code tries to grab up to 100 (MAXDEVICES) GPUs.
  // If there are no GPUs, it will get a default device.
  //
#define MAXDEVICES 100

  sycl::queue myQueues[MAXDEVICES];
  int ndevs = 0;
  try {
    auto P = sycl::platform(sycl::gpu_selector_v);
    auto RootDevices = P.get_devices();
    // auto C = sycl::context(RootDevices);
    for (auto &D : RootDevices) {
      myQueues[ndevs++] = sycl::queue(D, sycl::property::queue::enable_profiling{});
      if (ndevs >= MAXDEVICES)
	break;
    }
  } 
  catch (sycl::exception e) {
    ndevs = 1;
    myQueues[0] = sycl::queue(sycl::property::queue::enable_profiling{});
  }

  ndevs = arg_ndevs;

  try {

  auto inImgWidth = inImage.width();
  auto inImgHeight = inImage.height();
  auto channels = inImage.channels();
  auto filterWidth = filter.width();
  /* The halo is a region of extra space added to the image to make sure that the filter does not extend outside the image */
  auto halo = filter.half_width();

  /* Defines the iteration domain of the entire image */
  auto globalRange = sycl::range(inImgWidth, inImgHeight);
  /* Local thread group work size */
  auto localRange = sycl::range(1, 32);
  /* ndRange object that lets us access both the global and the local ranges*/
  auto ndRange = sycl::nd_range(globalRange, localRange);

  /* The multiplication operator * returns a range object with the same dimensionality (2) in this case,
   * and it does an elementwise multiplication of the two ranges. 
   * In this case, inBufRange appears to be an iteration domain slightly larger than the entire image,
   * and the multiplication operator makes it so there is an entry in the range for each of the 3 color channels (RGB).
   */
  auto inBufRange =
      sycl::range(inImgHeight + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
  /* What both of these ranges are is similar. Both are just ranges expanded so there is one element for each color. */
  auto outBufRange =
      sycl::range(inImgHeight, inImgWidth) * sycl::range(1, channels);

  auto filterRange = filterWidth * sycl::range(1, channels);

#ifdef MYDEBUGS
  std::cout << "inImgWidth: " << inImgWidth << "\ninImgHeight: " << inImgHeight
            << "\nchannels: " << channels << "\nfilterWidth: " << filterWidth
            << "\nhalo: " << halo << "\n";
#endif

  /* PARALLEL IMPLEMENTATION BEGINS HERE */

  /* Partition image across the 4 devices row-wise */
  int inImgPartitionWidth = inImgWidth;
  int inImgPartitionHeight = inImgHeight / ndevs;

  sycl::range<2> partitionGlobalRange = sycl::range(inImgPartitionWidth, inImgPartitionHeight);
  sycl::nd_range<2> partitionNdRange = sycl::nd_range(partitionGlobalRange,localRange);

  auto partitionInBufRange = 
        sycl::range(inImgPartitionHeight + (halo * 2), inImgPartitionWidth + (halo * 2)) *
        sycl::range(1, channels);
  auto partitionOutBufRange =
      sycl::range(inImgPartitionHeight, inImgPartitionWidth) * sycl::range(1, channels);
{

    /* Offsets used to compute partition for each device */
    size_t offsetOut = (outImage.height() / ndevs) * outImage.width() * outImage.channels(); 
    size_t offsetIn = (inImage.height() / ndevs) * inImage.width_with_halo() * inImage.channels();
    
    /* Filterbuf is the same for all queues */
    auto filterBuf = sycl::buffer{filter.data(), filterRange};
    
    assert(ndevs<=4);
    sycl::event e1, e2, e3, e4;
    std::vector<sycl::event> events = {e1, e2, e3, e4};

    std::vector<sycl::buffer<float, 2>> outBufParts, inBufParts;
    for (int i=0; i<ndevs; i++) {
        size_t curOffsetOut = offsetOut*i;
	auto outBufPart = sycl::buffer<float, 2>{partitionOutBufRange};
	outBufPart.set_final_data(outImage.data() + curOffsetOut);
	outBufParts.push_back(outBufPart);

        size_t curOffsetIn = offsetIn*i;
        sycl::buffer<float, 2> inBufPart = sycl::buffer<float, 2>{inImage.data() + curOffsetIn , partitionInBufRange};
	inBufParts.push_back(inBufPart);
    }

#ifdef PROFILE
    auto stime = std::chrono::system_clock::now();
#endif
    for (int queueId=0; queueId<ndevs; queueId++) {

        /* Create buffers from offsets */
	auto inBufPart = inBufParts[queueId];
	auto outBufPart = outBufParts[queueId];
        
        /* Submit kernels on each device */
        sycl::queue queue = myQueues[queueId];
        events[queueId] = queue.submit([&](sycl::handler& cgh) {

            sycl::accessor filterAccessor{filterBuf, cgh};
            sycl::accessor inBufAccessor{inBufPart, cgh};
            sycl::accessor outBufAccessor{outBufPart, cgh};

            cgh.parallel_for(partitionNdRange, [=](sycl::nd_item<2> item) {
                sycl::id<2> globalId = item.get_global_id();
                globalId = sycl::id{globalId[1], globalId[0]};

                auto channelsStride = sycl::range(1, channels);
                auto haloOffset = sycl::id(halo, halo);
                auto src = (globalId + haloOffset) * channelsStride;
                auto dest = globalId * channelsStride;


                // 100 is a hack - so the dim is not dynamic
                float sum[/* channels */ 100];
                assert(channels < 100);

                
                for (size_t i = 0; i < channels; ++i) {
                  sum[i] = 0.0f;
                }

		for (int k=0; k<100; k++){
                for (int r = 0; r < filterWidth; ++r) {
                  for (int c = 0; c < filterWidth; ++c) {
                    auto srcOffset =
                    sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
                    auto filterOffset = sycl::id(r, c * channels);

                    for (int i = 0; i < channels; ++i) {
                      auto channelOffset = sycl::id(0, i);
                      sum[i] += inBufAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
                    }
                  }
                }
		}

                for (size_t i = 0; i < channels; ++i) {
                  outBufAccessor[dest + sycl::id{0, i}] = sum[i];
                }
            });
        });
    }

    /* Wait for all devices to finish */
    for (int queueId=0; queueId<ndevs; queueId++) {
        myQueues[queueId].wait_and_throw();
#ifdef PROFILE
        //sycl::event e = events[queueId];
        //double multiGPUTime = e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
	    //		  e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    	//std::cout<< "Runtime on GPU "<<queueId<<": "<< multiGPUTime
          //    << " nanoseconds (" << multiGPUTime / 1.0e9 << " seconds)\n";
#endif
    }
#ifdef PROFILE
    auto etime = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(etime - stime).count();
    std::cout<<"Total runtime on "<<ndevs<<" GPUs: "<<diff
        <<" nanoseconds (" <<diff / 1.0e9 << " seconds)"<<std::endl;
#endif
  }
}
catch (sycl::exception e) {
  std::cout << "Exception caught: " << e.what() << std::endl;
}


util::write_image(outImage, outFile);
}
 
