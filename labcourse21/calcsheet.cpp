#include <iostream>
#include <cmath>
#include <vector>
#include "autodiff/forward/dual.hpp"
#include "autodiff/forward/dual/eigen.hpp"

using namespace autodiff;

// Class for error propagation with arbitrary number of input variables
class ErrorPropagation {
public:
    // Constructor
    ErrorPropagation() {}

    // Method to calculate propagated error for a given function and uncertainties
    template <typename Func, typename... Args>
    double calculate(Func&& func, const std::tuple<Args...>& values, const std::vector<double>& uncertainties) {
        // Unpack the values and uncertainties
        auto unpacked_values = unpack_values(values);
        
        // Evaluate the function
        dual z = func(unpacked_values);

        // Calculate the partial derivatives
        std::vector<double> partial_derivatives;
        partial_derivatives = compute_partials(func, unpacked_values);

        // Gaussian error propagation formula
        double sigma_z = 0.0;
        for (size_t i = 0; i < uncertainties.size(); ++i) {
            sigma_z += std::pow(partial_derivatives[i] * uncertainties[i], 2);
        }
        sigma_z = std::sqrt(sigma_z);

        // Output results
        //std::cout << "Function value (z): " << z << std::endl;
        //std::cout << "Partial derivatives: ";
        //for (auto& pd : partial_derivatives) {
        //    std::cout << pd << " ";
        // }
        //std::cout << std::endl;
        //std::cout << "Propagated uncertainty (σ_z): " << sigma_z << std::endl;

        return sigma_z;
    }

private:
    // Unpacks the tuple values into a vector (to simplify usage in variadic functions)
    template <typename Tuple, std::size_t... I>
    auto unpack_values_impl(const Tuple& t, std::index_sequence<I...>) {
        return std::make_tuple(std::get<I>(t)...);
    }

    template <typename Tuple>
    auto unpack_values(const Tuple& t) {
        return unpack_values_impl(t, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
    }

    // Compute the partial derivatives with respect to each variable
    template <typename Func, typename... Args>
    std::vector<double> compute_partials(Func&& func, const std::tuple<Args...>& values) {
        std::vector<double> partials;
        
        // Loop through each variable and compute the partial derivative
        size_t index = 0;
        ((partials.push_back(derivative(func, wrt(std::get<index>(values)), at(values)))), ++index);
        
        return partials;
    }
};


// Scale factor vf with the two measured lengths la and lb, taken from the points C and D
dual scalefactor(dual la, dual lb) {
    return (3 * la - lb) / (2 * 30.01); // 30.01cm as standard reference
}

// Radius r, explicit with measured radius rm and scale factor vf
dual radius(dual rm, dual vf) {
	return rm / vf;
}

// Momentum p with magnetic field strength B = 2.05 T, radius r (scaled)
dual momentum(dual r) {
	return 2.998 * 2.05 * r;
}

int main() {
    // scale factor parameters
	// Lengths in cm
    double la_value = 45.5;
    double lb_value = 59.0;

    // scale factor uncertainty values
    double sigma_la = 0.1;
    double sigma_lb = 0.1;

	// radius parameters
	// rm in cm
	double rm_value = 230;
	// vf_value from function scalefactor
	
	//radius uncertainty values
	double rm_sigma = 7.5;
	// vf_sigma from error propagation scalefactor

    // Create an instance of the ErrorPropagation class
    ErrorPropagation error_propagation;

    // Call the calculate method with example_function and the input values and uncertainties
    sigma_vf = error_propagation.calculate(vf, std::make_tuple(la_value, lb_value), {sigma_la, sigma_lb});
	std::cout << "Propagated uncertainty (σ_vf): " << sigma_vf << std::endl;
    return 0;
}

