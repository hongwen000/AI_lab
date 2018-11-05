#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "decision_tree.h"
// Include the headers of MyLib

namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(pyMyLib)
{
    Py_Initialize();
    np::initialize();

}