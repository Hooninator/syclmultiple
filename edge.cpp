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
 
#define MYDEBUGS
#define DEBUGDUMP
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

  if (argc == 2) {
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

#ifdef CORRECTNESS
  auto outImageCorrect = util::allocate_image(inImage.width(), inImage.height(),
                                        inImage.channels());
#endif 
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
      myQueues[ndevs++] = sycl::queue(D,sycl::property::queue::enable_profiling{});
      if (ndevs >= MAXDEVICES)
	break;
    }
  } 
  catch (sycl::exception e) {
    ndevs = 1;
    myQueues[0] = sycl::queue(sycl::property::queue::enable_profiling{});
  }

  try {
    /* Define queues */
    sycl::queue myQueue1 = myQueues[0];

#ifdef MYDEBUGS
    auto t1 = std::chrono::steady_clock::now();  // Start timing
#endif


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
    size_t offsetOut = (outImage.height() / ndevs) * outImage.width(); 
    size_t offsetIn = (inImage.height_with_halo() / ndevs) * outImage.width_with_halo();
    
    /* Filterbuf is the same for all queues */
    auto filterBuf = sycl::buffer{filter.data(), filterRange};

    std::cout<<outImage.data()[offsetOut]<<std::endl;

    for (int queueId=0; queueId<ndevs; queueId++) {
        size_t curOffsetOut = offsetOut*queueId;
        size_t curOffsetIn = offsetIn*queueId;

	std::cout<<"Out offset: "<<curOffsetOut<<std::endl;
        /* Create buffers from offsets */
        sycl::buffer<float, 2> inBufPart = sycl::buffer<float, 2>{inImage.data() + curOffsetIn , partitionInBufRange};
        sycl::buffer<float, 2> outBufPart = sycl::buffer<float, 2>{partitionOutBufRange};

	/* Write to appropriate region of output image */
        outBufPart.set_final_data(outImage.data() + curOffsetOut);
        
        /* Submit kernels on each device */
        sycl::queue queue = myQueues[queueId];
        sycl::event e = queue.submit([&](sycl::handler& cgh) {
            sycl::accessor filterAccessor{filterBuf, cgh, sycl::read_only};
            sycl::accessor inBufAccessor{inBufPart, cgh, sycl::read_only};
            sycl::accessor outBufAccessor{outBufPart, cgh, sycl::write_only};
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

		for (size_t i = 0; i < channels; ++i) {
		  outBufAccessor[dest + sycl::id{0, i}] = sum[i];
		}

            });
        });
#ifdef PROFILE
	double multiGPUTime = e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
	    		  e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    	std::cout<< "Runtime on GPU "<<queueId<<": "<< multiGPUTime
              << " nanoseconds (" << multiGPUTime / 1.0e9 << " seconds)\n";
#endif
    }

    /* Wait for all devices to finish */
    for (int i=0; i<ndevs; i++)
        myQueues[i].wait();

#ifdef CORRECTNESS
    /* Rerun the image blurring on a single device to make sure it works */
    auto inBuf = sycl::buffer{inImage.data(), inBufRange};
    auto outBuf = sycl::buffer<float, 2>{outBufRange};
    outBuf.set_final_data(outImageCorrect.data());

    sycl::event e1 = myQueue1.submit([&](sycl::handler& cgh1) {
      sycl::accessor inAccessor{inBuf, cgh1, sycl::read_only};
      sycl::accessor outAccessor{outBuf, cgh1, sycl::write_only};
      sycl::accessor filterAccessor{filterBuf, cgh1, sycl::read_only};

      cgh1.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
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

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset =
                sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            for (int i = 0; i < channels; ++i) {
              auto channelOffset = sycl::id(0, i);
              sum[i] += inAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        for (size_t i = 0; i < channels; ++i) {
          outAccessor[dest + sycl::id{0, i}] = sum[i];
        }
      });
    });

    myQueue1.wait_and_throw();


#endif

#ifdef CORRECTNESS
#ifdef PROFILE
    double singleGPUTime = (e1.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e1.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());
    std::cout << "Single GPU runtime: " << singleGPUTime
              << " nanoseconds (" << singleGPUTime / 1.0e9 << " seconds)\n";
#endif
#endif
  }
}
catch (sycl::exception e) {
  std::cout << "Exception caught: " << e.what() << std::endl;
}

#ifdef CORRECTNESS
check_image_correct(outImage, outImageCorrect);
#endif

util::write_image(outImage, outFile);
}
 
