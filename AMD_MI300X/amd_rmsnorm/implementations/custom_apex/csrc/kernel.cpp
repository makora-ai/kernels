#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


namespace {

void compute_n1_n2(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    int& n1,
    int& n2)
{
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
        assert( input.sizes()[i+idiff] == normalized_shape[i] );
        n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
        n1 *= input.sizes()[i];
    }
}

void check_args(
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    at::Tensor beta
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
    TORCH_CHECK(!beta.defined() || beta.sizes().equals(normalized_shape));
}

void check_args(
    at::IntArrayRef normalized_shape,
    at::Tensor gamma
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
}


void check_args(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input,normalized_shape,n1,n2);
}

void check_args(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    at::Tensor beta,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma,beta);
}

void check_args(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma);
}

} // namespace {


void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    at::Tensor* residual,
    at::Tensor* inter_out,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon);


void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* inter_out,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_residual,
    at::Tensor* grad_gamma);


std::vector<at::Tensor> rms_norm(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon,
    bool training)
{
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    int n1,n2;
    check_args(input,normalized_shape,gamma,n1,n2);
    at::Tensor output = at::empty_like(input);
    const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();

    if (training) {
        at::Tensor invvar = at::empty({n1}, input.options().dtype(stats_dtype));
        cuda_rms_norm(&output, &invvar, &input, NULL, NULL, n1, n2, normalized_shape, &gamma, epsilon);
        return { output, invvar };
    } else {
        cuda_rms_norm(&output, NULL, &input, NULL, NULL, n1, n2, normalized_shape, &gamma, epsilon);
        return { output };
    }
}


std::vector<at::Tensor> rms_norm_residual(
    at::Tensor input,
    at::Tensor residual,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon,
    bool training)
{
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    int n1,n2;
    check_args(input,normalized_shape,gamma,n1,n2);
    at::Tensor output = at::empty_like(input);
    at::Tensor inter_out = at::empty_like(input);
    const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();

    if (training) {
        at::Tensor invvar = at::empty({n1}, input.options().dtype(stats_dtype));
        cuda_rms_norm(&output, &invvar, &input, &residual, &inter_out, n1, n2, normalized_shape ,&gamma, epsilon);
        return { output, invvar, inter_out };
    } else {
        cuda_rms_norm(&output, NULL, &input, &residual, NULL, n1, n2, normalized_shape ,&gamma, epsilon);
        return { output };
    }
}



std::vector<at::Tensor> rms_norm_backward(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor input_or_output,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon)
{
    CHECK_INPUT(dout);
    CHECK_INPUT(invvar);
    CHECK_INPUT(input_or_output);
    CHECK_INPUT(gamma);
    int n1,n2;
    check_args(input_or_output,normalized_shape,gamma,n1,n2);
    at::Tensor grad_input = at::empty_like(input_or_output);
    at::Tensor grad_gamma = at::empty_like(gamma);
    cuda_rms_norm_gradient(
        &dout,
        &invvar,
        &input_or_output,
        n1,
        n2,
        normalized_shape,
        &gamma,
        epsilon,
        &grad_input,
        NULL,
        &grad_gamma);

    return { grad_input, grad_gamma };
}


std::vector<at::Tensor> rms_norm_residual_backward(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor inter_out,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon)
{
    CHECK_INPUT(dout);
    CHECK_INPUT(inter_out);
    CHECK_INPUT(gamma);
    int n1,n2;
    check_args(inter_out,normalized_shape,gamma,n1,n2);
    at::Tensor grad_input = at::empty_like(inter_out);
    at::Tensor grad_residual = at::empty_like(inter_out);
    at::Tensor grad_gamma = at::empty_like(gamma);
    cuda_rms_norm_gradient(
        &dout,
        &invvar,
        &inter_out,
        n1,
        n2,
        normalized_shape,
        &gamma,
        epsilon,
        &grad_input,
        &grad_residual,
        &grad_gamma);

    return { grad_input, grad_residual, grad_gamma };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, "RMSNorm forward (CUDA)", py::call_guard<py::gil_scoped_release>());
    m.def("rms_norm_residual", &rms_norm_residual, "RMSNorm forward with residual addition (CUDA)", py::call_guard<py::gil_scoped_release>());
    m.def("rms_norm_backward", &rms_norm_backward, "RMSNorm backward (CUDA)", py::call_guard<py::gil_scoped_release>());
    m.def("rms_norm_residual_backward", &rms_norm_residual_backward, "RMSNorm backward with residual addition (CUDA)", py::call_guard<py::gil_scoped_release>());
}
