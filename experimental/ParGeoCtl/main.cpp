#include <iostream>
#include <unistd.h>
#include <limits>
#include "json.hpp"

using json = nlohmann::json;

json create_response(std::string status, double time, std::vector<double> result){

    json response;
    response["status"] = status;
    response["time"] = time;
    response["result"] = result;
    return response;
}

int main(int argc, char *argv[]) {

    json message;
    json response;
    std::string message_str;
    std::vector<double> empty_vector;

    while(true) {
        std::getline(std::cin, message_str);
        //std::cout<<"Received:"<<message_str<<std::endl;

        message = json::parse(message_str.c_str());
        std::string message_type = message["type"].template get<std::string>();

        if(message_type.compare("run-query") == 0) {

            // The query radius
            double radius = message["radius"].template get<double>();

            // Read the weights into a vector
            // TODO: Use these to run the query.
            std::vector<double> weights = message["weights"].template get<std::vector<double>>();

            // std::cout << "\tweights: " << std::endl;
            // for (double w : weights) {
            //     std::cout << "\t\t" << w << std::endl;
            // }

            // Generate response
            // TODO: Update with actual result.
            std::vector<double> query_result {0.001, 0.02};
            response = create_response("OK", 0.001, query_result);
            std::cout << response.dump() << std::endl;
        }
        else if(message_type.compare("build-datastructure") == 0) {

            // Get the dimension
            int dimension = message["dimension"].template get<int>();
            // std::cout << "\tdimension: " << dimension << std::endl;

            // Read the points into a vector.
            // TODO: Use these to build the data-structure
            std::vector<std::vector<double>> points = message["points"].template get<std::vector<std::vector<double>>>();

            // std::cout << "\tpoints: " << std::endl;
            // for (std::vector<double> p : points) {
            //     std::cout << "\t\t" << p[0] << ", " << p[1] << ", " << p[2] << std::endl;
            // }

            // TODO: Do any error handling required from ParGeo.


            response = create_response("OK", 0.001, empty_vector);
            std::cout << response.dump() << std::endl;
        }
        else if(message_type.compare("exit") == 0) {
            exit(0);
        }
        else {
            response = create_response("ERROR", 0, empty_vector);
            std::cout << response.dump() << std::endl;
        }
    }
    return 0;
}