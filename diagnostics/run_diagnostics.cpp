import tests.correctness;
import tests;
#include <iostream>
#include<vector>
#include<sstream>
#include <iomanip>
#include <ostream>


bool run_correctness_tests(std::ostream& os, bool save_metrics){
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y_%Hh-%Mm-%Ss");
	

	std::vector<desal::test::CorrectnessTest> v;

	//GPU Diagnostics
	#ifdef opt_use_cuda
			//v.push_back(desal::test::CorrectnessTest("reduce_sum_f32_device_ascending",desal::test::corr::reduce_sum_f32_device_ascending));
			//v.push_back(desal::test::CorrectnessTest("reduce_sum_f32_device_descending",desal::test::corr::reduce_sum_f32_device_ascending));
			//v.push_back(desal::test::CorrectnessTest("reduce_sum_f64_device_ascending",desal::test::corr::reduce_sum_f64_device_ascending));
			//v.push_back(desal::test::CorrectnessTest("reduce_sum_f64_device_descending",desal::test::corr::reduce_sum_f64_device_ascending));

			//v.push_back(desal::test::CorrectnessTest("test_reduce_sum_of_squares_poisson_field_residual_f32_device_uniform",desal::test::corr::reduce_sum_of_squares_poisson_field_residual_f32_device_uniform));
			v.push_back(desal::test::CorrectnessTest("test_reduce_sum_of_squares_poisson_field_residual_f32_device_uniform",desal::test::corr::mg_vc_poisson_2D_f32_zero_B));
			
	#endif
	
	bool result=true;
	
	for (int i=0; i<v.size(); i++){
		bool temp=v[i].run_test();
		if (temp!= true){
			result=temp;
		}
		
		if (v[i].test_successful){
			os<<v[i].name<<" finished successfully\n";
		}
		else{
			os<<v[i].name<<" finished erroneously\n";
		}
		
		if ((i%5)==0){
			os<<i+1<<" out of "<<v.size()<<" correctness tests finished \n";
		}
	}
	if (save_metrics==true){
		//desal::test::corr::save_metrics(v,oss.str());
	}
	
	return result;

}

bool run_tests(std::ostream& os,bool save_stats){

	//bool test1=run_performance_tests(os, save_stats); //run performance tests
	bool test2=run_correctness_tests(os,save_stats); //run correctness tests
	return test2;
}

int main(){
	bool save_stats=true; //save test logs to file
	bool test_result=run_tests(std::cout,save_stats);
	
	if (test_result){
		std::cout<<"All tests finished without errors\n";
		return 1;
	}
	else{
		std::cout<<"At least one test finished erroneously\n";
		return 0;
	}
	
	
}