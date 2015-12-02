#include <cstdio>
#include <time.h>
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskID {TOP_LEVEL_TASK_ID, INIT_DATA_TASK_ID, MERGE_SORT_TASK_ID, MERGE_TASK_ID, COPY_TASK_ID};

enum FieldIds {FID_FIELD_IO, FID_FIELD_WORK};

void init_data_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime)
{
	// Check that the inputs look right since we have no
	// static checking to help us out.
	assert(regions.size() == 1);
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 1);
	FieldID fid = *(task->regions[0].privilege_fields.begin());

	RegionAccessor<AccessorType::Generic, int> acc =
			regions[0].get_field_accessor(fid).typeify<int>();

	srand(time(NULL));

	// fill in with random numbers
	Domain dom = runtime->get_index_space_domain(ctx,
			task->regions[0].region.get_index_space());

	Rect<1> rect = dom.get_rect<1>();
	for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
	{
		acc.write(DomainPoint::from_point<1>(pir.p), rand());
	}
}

// copy between 2 logical regions from regions[0] to regions[1]
void copy_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime)
{
	Domain dom0 = runtime->get_index_space_domain(ctx,
			task->regions[0].region.get_index_space());
	Domain dom1 = runtime->get_index_space_domain(ctx,
			task->regions[1].region.get_index_space());

	Rect<1> rect0 = dom0.get_rect<1>();
	Rect<1> rect1 = dom1.get_rect<1>();

	FieldID fid0 = *(task->regions[0].privilege_fields.begin());
	FieldID fid1 = *(task->regions[1].privilege_fields.begin());

	RegionAccessor<AccessorType::Generic, int> acc0 =
			regions[0].get_field_accessor(fid0).typeify<int>();
	RegionAccessor<AccessorType::Generic, int> acc1 =
			regions[1].get_field_accessor(fid1).typeify<int>();

	for (GenericPointInRectIterator<1> pir0(rect0), pir1(rect1); pir0 && pir1; pir0++, pir1++)
	{
		acc1.write(DomainPoint::from_point<1>(pir1.p), acc0.read(DomainPoint::from_point<1>(pir0.p)));
	}
}

// TODO
void merge_sort_task_local(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime)
{

}

// a recursive task
void merge_sort_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime)
{
	// figure out input size
	Domain dom = runtime->get_index_space_domain(ctx,
			task->regions[0].region.get_index_space());
	Rect<1> rect = dom.get_rect<1>();
	const int DIM_SIZE = rect.dim_size(0);

	// to show progress on screen
	printf(".\n");

	if(DIM_SIZE <= 1)
	{
		return;
	}
	else
	{
		Rect<1> dataRect(Point<1>(0), Point<1>(DIM_SIZE-1));
		IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<1>(dataRect));

		// Somehow, I cannot partition the LR passed in to the task.
		// To workaround it, create a local logical partition with the same size of LR passed in
		FieldSpace ioFS = runtime->create_field_space(ctx);
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx, ioFS);
			allocator.allocate_field(sizeof(int), FID_FIELD_IO);
		}

		LogicalRegion ioLR = runtime->create_logical_region(ctx, is, ioFS);

		// Copy content from passed in region to local region
		{
			TaskLauncher cpLauncher(COPY_TASK_ID, TaskArgument(NULL, 0));
			cpLauncher.add_region_requirement(
					RegionRequirement(task->regions[0].region, READ_ONLY, EXCLUSIVE, task->regions[0].region));
			cpLauncher.add_field(0/*index*/, FID_FIELD_IO);
			cpLauncher.add_region_requirement(
					RegionRequirement(ioLR, WRITE_DISCARD, EXCLUSIVE, ioLR));
			cpLauncher.add_field(1/*index*/, FID_FIELD_IO);
			runtime->execute_task(ctx, cpLauncher);
		}

		Rect<1> color_bounds(Point<1>(0),Point<1>(1));
		Domain color_domain = Domain::from_rect<1>(color_bounds);
		DomainColoring coloring;
		IndexPartition ip0;

		if(DIM_SIZE % 2 != 0)
		{
			Rect<1> subrect0(Point<1>(0),Point<1>(DIM_SIZE/2-1));
			coloring[0] = Domain::from_rect<1>(subrect0);

			Rect<1> subrect1(Point<1>(DIM_SIZE/2),Point<1>(DIM_SIZE-1));
			coloring[1] = Domain::from_rect<1>(subrect1);

			// application which makes use of a non-disjoint partition.
			ip0 = runtime->create_index_partition(ctx, is, color_domain,
					coloring, true/*disjoint*/);
		}
		else
		{
		    Blockify<1> coloring(DIM_SIZE/2);
		    ip0 = runtime->create_index_partition(ctx, is, coloring);
		}

		// create partition on local LR
		LogicalPartition ioLP = runtime->get_logical_partition(ctx, ioLR, ip0);
		Domain launch_domain = color_domain;
		ArgumentMap arg_map;

		// dispatch sub task
		{
			IndexLauncher msLauncher(MERGE_SORT_TASK_ID, launch_domain,
					TaskArgument(NULL, 0), arg_map);

			msLauncher.add_region_requirement(
					RegionRequirement(ioLP, 0/*projection ID*/,
							READ_WRITE, EXCLUSIVE, ioLR));
			msLauncher.add_field(0, FID_FIELD_IO);

			runtime->execute_index_space(ctx, msLauncher);
		}

		// merge
		{
			TaskLauncher mergeLauncher(MERGE_TASK_ID, TaskArgument(NULL, 0));
			mergeLauncher.add_region_requirement(
					RegionRequirement(task->regions[0].region, READ_WRITE, EXCLUSIVE, task->regions[0].region));
			mergeLauncher.add_field(0/*index*/, FID_FIELD_IO);
			mergeLauncher.add_region_requirement(
					RegionRequirement(ioLR, READ_ONLY, EXCLUSIVE, ioLR));
			mergeLauncher.add_field(1/*index*/, FID_FIELD_IO);
			runtime->execute_task(ctx, mergeLauncher);
		}
	}

	// do sanity check: print out result in each iteration
	{
		Domain dom = runtime->get_index_space_domain(ctx,
				task->regions[0].region.get_index_space());

		Rect<1> rect = dom.get_rect<1>();

		FieldID fid0 = *(task->regions[0].privilege_fields.begin());
		RegionAccessor<AccessorType::Generic, int> acc =
				regions[0].get_field_accessor(fid0).typeify<int>();

		printf("[ ");
		int prev = 0;
		int current = 0;
		bool all_passed = true;
		for(GenericPointInRectIterator<1> pir(rect);pir; pir++)
		{
			current = acc.read(DomainPoint::from_point<1>(pir.p));
			printf("%d ", current);
			if(int(pir.p) != int(rect.lo))
			{
				if(current < prev)
				{
					all_passed = false;
				}
			}
			prev = current;
		}
		printf("] ");
		printf("All Passed ? %s\n", all_passed? "Yes": "No");
	}
}

// merge: it takes 2 regions, regions[0] for writing back, regions[1] contains 2 sorted partitions
void merge_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime)
{
	Domain dom = runtime->get_index_space_domain(ctx,
			task->regions[0].region.get_index_space());

	Rect<1> rect = dom.get_rect<1>();

	Domain dom2 = runtime->get_index_space_domain(ctx,
			task->regions[1].region.get_index_space());

	Rect<1> rect2 = dom2.get_rect<1>();

	const int DIM_SIZE1 = rect2.dim_size(0);

	GenericPointInRectIterator<1> pir0(rect);
	GenericPointInRectIterator<1> pir1Left(rect2);
	GenericPointInRectIterator<1> pir1LeftEnd(rect2);
	GenericPointInRectIterator<1> pir1Right(rect2);

	for(int i = 0; i<(DIM_SIZE1/2); i++)
	{
		// figure out the starting point of right hand side partition
		pir1Right++;
		// upper bound of the left hand side partition
		pir1LeftEnd++;
	}

	FieldID fid0 = *(task->regions[0].privilege_fields.begin());
	FieldID fid1 = *(task->regions[1].privilege_fields.begin());
	RegionAccessor<AccessorType::Generic, int> acc0 =
			regions[0].get_field_accessor(fid0).typeify<int>();
	RegionAccessor<AccessorType::Generic, int> acc1 =
			regions[1].get_field_accessor(fid1).typeify<int>();

	while((int(pir1Left.p) < int(pir1LeftEnd.p)) && pir1Right)
	{
		int lElement = acc1.read(DomainPoint::from_point<1>(pir1Left.p));
		int rElement = acc1.read(DomainPoint::from_point<1>(pir1Right.p));

		if(lElement <= rElement)
		{
			acc0.write(DomainPoint::from_point<1>(pir0.p), lElement);
			pir1Left++;
		}
		else
		{
			acc0.write(DomainPoint::from_point<1>(pir0.p), rElement);
			pir1Right++;
		}

		pir0++;
	}

	while((int(pir1Left.p) < int(pir1LeftEnd.p)))
	{
		acc0.write(DomainPoint::from_point<1>(pir0.p), acc1.read(DomainPoint::from_point<1>(pir1Left.p)));
		pir1Left++;
		pir0++;
	}

	while(pir1Right)
	{
		acc0.write(DomainPoint::from_point<1>(pir0.p), acc1.read(DomainPoint::from_point<1>(pir1Right.p)));
		pir1Right++;
		pir0++;
	}
}

void top_level_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime)
{
	int dataSize = 10;
	int c = 0;
	int partitionNumber = 1;
	const InputArgs &command_args = HighLevelRuntime::get_input_args();
	if(command_args.argc > 1)
	{
		// logic to handle options
		while((c = getopt(command_args.argc, command_args.argv, "n:p:")) != -1)
		{
			switch(c)
			{
			case 'n':
				dataSize = atoi(optarg);
				break;
			case 'p':
				partitionNumber = atoi(optarg);
				break;
			default:
				break;
			}
		}
	}

	if(partitionNumber <= 0)
	{
		partitionNumber = 1;
	}

	// index space of data
	Rect<1> dataRect(Point<1>(0), Point<1>(dataSize-1));
	IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<1>(dataRect));
	runtime->attach_name(is, "is");

	// field space
	FieldSpace ioFS = runtime->create_field_space(ctx);
	runtime->attach_name(ioFS, "ioFS");
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx, ioFS);
		allocator.allocate_field(sizeof(int), FID_FIELD_IO);
		runtime->attach_name(ioFS, FID_FIELD_IO, "FID_FIELD_IO");
	}

	// logical region
	LogicalRegion ioLR = runtime->create_logical_region(ctx, is, ioFS);
	runtime->attach_name(ioLR, "ioLR");

	// launch initialization task
	{
		TaskLauncher initLauncher(INIT_DATA_TASK_ID, TaskArgument(&dataSize, sizeof(dataSize)));
		initLauncher.add_region_requirement(
				RegionRequirement(ioLR, WRITE_DISCARD, EXCLUSIVE, ioLR));
		initLauncher.add_field(0/*index*/, FID_FIELD_IO);
		runtime->execute_task(ctx, initLauncher);
	}
	// launch main merge sort task
	{
		TaskLauncher msLauncher(MERGE_SORT_TASK_ID, TaskArgument());
		msLauncher.add_region_requirement(
				RegionRequirement(ioLR, READ_WRITE, EXCLUSIVE, ioLR));
		msLauncher.add_field(0/*index*/, FID_FIELD_IO);
		runtime->execute_task(ctx, msLauncher);
	}
}

int main(int argc, char **argv)
{
	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
			Processor::LOC_PROC, true /*single*/, false/*index*/);
	HighLevelRuntime::register_legion_task<init_data_task>(INIT_DATA_TASK_ID,
			Processor::LOC_PROC, true /*single*/, false/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "init_data_task");
	HighLevelRuntime::register_legion_task<merge_sort_task>(MERGE_SORT_TASK_ID,
			Processor::LOC_PROC, true /*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "merge_sort_task");
	HighLevelRuntime::register_legion_task<merge_task>(MERGE_TASK_ID,
			Processor::LOC_PROC, true /*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "sort_task");
	HighLevelRuntime::register_legion_task<copy_task>(COPY_TASK_ID,
				Processor::LOC_PROC, true /*single*/, true/*index*/,
				AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "copy_task");

	return HighLevelRuntime::start(argc, argv);
}
