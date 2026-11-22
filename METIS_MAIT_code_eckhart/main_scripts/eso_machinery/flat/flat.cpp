// This file is part of the PyHDRL Python language bindings
// Copyright (C) 2023-2026 European Southern Observatory
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "hdrlfunc/flat.hpp"

#include <memory>

#include <cpl_image_io.h>
#include <hdrl_flat.h>
#include <hdrl_image.h>
#include <pybind11/pybind11.h>

#include "hdrlcore/error.hpp"
#include "hdrlcore/image.hpp"

namespace hdrl
{
namespace func
{
using hdrl::core::Error;
using hdrl::core::Image;
using hdrl::core::pycpl_image;

Flat::Flat(cpl_size filter_size_x, cpl_size filter_size_y,
           hdrl_flat_method method)
{
  m_interface = Error::throw_errors_with(hdrl_flat_parameter_create,
                                         filter_size_x, filter_size_y, method);
}

// TODO: could make stat_mask an optional arg in the bindings
py::object
Flat::compute(std::shared_ptr<ImageList> hdrl_data, Collapse collapse,
              pycpl_mask stat_mask)
{
  hdrl_image* master;
  cpl_image* contrib_map;
  Error::throw_errors_with(hdrl_flat_compute, hdrl_data.get()->ptr(),
                           stat_mask.m, collapse.ptr(), m_interface, &master,
                           &contrib_map);
  /// return results in namedtuple FlatResult
  py::module_ col = py::module_::import("collections");
  py::object ntup = col.attr("namedtuple")("FlatResult", "master contrib_map");
  Image item = Image(master);
  py::object result =
      ntup(std::make_shared<Image>(item), pycpl_image(contrib_map));
  return result;
}

cpl_size
Flat::get_filter_size_x()
{
  return Error::throw_errors_with(hdrl_flat_parameter_get_filter_size_x,
                                  m_interface);
}

cpl_size
Flat::get_filter_size_y()
{
  return Error::throw_errors_with(hdrl_flat_parameter_get_filter_size_y,
                                  m_interface);
}

hdrl_flat_method
Flat::get_method()
{
  return Error::throw_errors_with(hdrl_flat_parameter_get_method, m_interface);
}

hdrl_parameter*
Flat::ptr()
{
  return m_interface;
}

}  // namespace func
}  // namespace hdrl
