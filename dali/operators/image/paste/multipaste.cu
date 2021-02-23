// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "dali/operators/image/paste/multipaste.h"
#include "dali/kernels/imgproc/paste/paste_gpu.h"
#include "dali/core/tensor_view.h"

namespace dali {

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteGPU, GPU)

bool MultiPasteGPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                              const workspace_t<GPUBackend> &ws) {
  AcquireArguments(spec_, ws);
  FillGPUInput(ws);

  const auto &images = ws.template InputRef<GPUBackend>(0);
  const auto &output = ws.template OutputRef<GPUBackend>(0);
  output_desc.resize(1);

  TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
            using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
            kernel_manager_.Initialize<Kernel>();

            TensorListShape<> sh = images.shape();
            TensorListShape<3> shapes(sh.num_samples(), sh.sample_dim());
            for (int i = 0; i < sh.num_samples(); i++) {
                const TensorShape<3> &out_sh = { output_size_[i].data[0],
                                                output_size_[i].data[1], sh[i][2] };
                shapes.set_tensor_shape(i, out_sh);
            }

            kernels::KernelContext ctx;
            ctx.gpu.stream = ws.stream();
            const auto tvin = view<const InputType, 3>(images);
            const auto &reqs = kernel_manager_.Setup<Kernel>(0, ctx, tvin,
                                                             samples, grid_cells, shapes);

            output_desc[0] = {shapes, TypeTable::GetTypeInfo(output_type_)};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
  return true;
}


void MultiPasteGPU::FillGPUInput(const workspace_t<GPUBackend> &ws) {
  auto &output = ws.template Output<GPUBackend>(0);
  auto out_shape = output.shape();
  int batch_size = out_shape.num_samples();
  for (int i = 0; i < batch_size; i++) {
    // Get all significant points on x and y axis - those will be the starts and ends of the cells
    int NO_DATA = 0;
    map<int, int> x_points;
    map<int, int> y_points;
    x_points[0] = NO_DATA;
    x_points[x_size] = NO_DATA;
    y_points[0] = NO_DATA;
    y_points[y_size] = NO_DATA;
    for (int i = 0; i < n; i++) {
      x_points[out_x_anchors[i]] = NO_DATA;
      x_points[out_x_anchors[i] + x_paste_size[i]] = NO_DATA;
      y_points[out_y_anchors[i]] = NO_DATA;
      y_points[out_y_anchors[i] + y_paste_size[i]] = NO_DATA;
    }

    // When we know how many of those points there are, we know how big our grid is for this output
    int x_grid_size = x_points.size() - 1;
    int y_grid_size = y_points.size() - 1;

    // Now lets fill forward and backward mapping of those significant points to grid cell indices
    vector<int> scaled_x_to_x;
    for (auto i = x_points.begin(); i != x_points.end(); i++) {
      i->second = scaled_x_to_x.size();
      scaled_x_to_x.push_back(i->first);
    }
    vector<int> scaled_y_to_y;
    for (auto i = y_points.begin(); i != y_points.end(); i++) {
      i->second = scaled_y_to_y.size();
      scaled_y_to_y.push_back(i->first);
    }

    // We create events that will fire when sweeping
    vector<vector<tuple<int, int, int>>> y_starting(y_grid_size + 1);
    vector<vector<tuple<int, int, int>>> y_ending(y_grid_size + 1);
    for (int i = 0; i < n; i++) {
      y_starting[y_points[out_y_anchors[i]]].emplace_back(
              i, x_points[out_x_anchors[i]], x_points[out_x_anchors[i] + x_paste_size[i]]);
      y_ending[y_points[out_y_anchors[i] + y_paste_size[i]]].emplace_back(
              i, x_points[out_x_anchors[i]], x_points[out_x_anchors[i] + x_paste_size[i]]);
    }
    y_starting[0].emplace_back(-1, 0, x_grid_size);
    y_ending[y_grid_size].emplace_back(-1, 0, x_grid_size);

    // And now the sweeping itself
    vector<GridCellInput> cells(x_grid_size * y_grid_size);
    vector<unordered_set<int>> starting(x_grid_size + 1);
    vector<unordered_set<int>> ending(x_grid_size + 1);
    set<int> open_pastes;
    for (int y = 0; y < y_grid_size; y++) {
      // Add open and close events on x axis for regions with given y start coordinate
      for (auto i = y_starting[y].begin(); i != y_starting[y].end(); i++) {
        starting[get<1>(*i)].insert(get<0>(*i));
        ending[get<2>(*i)].insert(get<0>(*i));
      }
      // Now sweep through x
      for (int x = 0; x < x_grid_size; x++) {
        // Open regions starting here
        for (auto i = starting[x].begin(); i != starting[x].end(); i++) {
          open_pastes.insert(*i);
        }

        // Take top most region
        int max_paste = *(--open_pastes.end());
        GridCellInput& cell = cells[y * x_grid_size + x];

        // And fill grid cell
        cell.cell_start[0] = scaled_y_to_y[y];
        cell.cell_start[1] = scaled_x_to_x[x];
        cell.cell_end[0] = scaled_y_to_y[y + 1];
        cell.cell_end[1] = scaled_x_to_x[x + 1];
        cell.input_idx = max_paste == -1 ? -1 : out_idx[max_paste];

        if (max_paste != -1) {
          cell.in_anchor[0] = in_y_anchors[max_paste] + cell.cell_start[0] - out_y_anchors[max_paste];
          cell.in_anchor[1] = in_x_anchors[max_paste] + cell.cell_start[1] - out_x_anchors[max_paste];
        }

        // Now remove regions that end here
        for (auto i = ending[x + 1].begin(); i != ending[x + 1].end(); i++) {
          open_pastes.erase(*i);
        }
      }
      // And remove start/events for regions whose y ends here
      for (auto i = y_ending[y + 1].begin(); i != y_ending[y + 1].end(); i++) {
        starting[get<1>(*i)].erase(get<0>(*i));
        ending[get<2>(*i)].erase(get<0>(*i));
      }
    }
  }
}

template<typename InputType, typename OutputType>
void MultiPasteGPU::RunImplExplicitlyTyped(workspace_t<GPUBackend> &ws) {
  const auto &images = ws.template Input<GPUBackend>(0);
  auto &output = ws.template Output<GPUBackend>(0);

  output.SetLayout(images.GetLayout());
  auto out_shape = output.shape();
  using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
  auto in_view = view<const InputType, 3>(images);
  auto out_view = view<OutputType, 3>(output);

  kernels::KernelContext ctx;
  kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, out_view, in_view, samples, grid_cells);
}


void MultiPasteGPU::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto input_type_id = ws.template InputRef<GPUBackend>(0).type().id();
  TYPE_SWITCH(input_type_id, type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
              RunImplExplicitlyTyped<InputType, OutputType>(ws);
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input_type_id)))  // NOLINT
}

}  // namespace dali
