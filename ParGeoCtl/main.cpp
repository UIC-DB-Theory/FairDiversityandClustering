#include <iostream>
#include <unistd.h>
#include "json.hpp"

using json = nlohmann::json;

const int READ_FROM_FILE = 100;
const int READ_JSON_STDIN = 200;

int main(int argc, char *argv[]) {
    
    int mode;
    json message;
    
    int opt;
    while((opt = getopt(argc, argv, ":j:f:")) != -1) 
    { 
        switch(opt) 
        { 
            // Read the message from a file
            case 'f': 
                mode =  READ_FROM_FILE;
                break;

            // Read the message from the command line
            case 'j':
                mode = READ_JSON_STDIN;
                message =  json::parse(optarg);
                break; 
        } 
    }

    // First message should be of type build-datastructure
    // For now, just printing the message details
    std::cout << "\ttype: " << message["type"].template get<std::string>() << std::endl;
    std::cout << "\tdimension: " << message["dimension"].template get<int>() << std::endl;
    // Read the points into a vector
    std::vector<std::vector<double>> points = message["points"].template get<std::vector<std::vector<double>>>();

    std::cout << "\tpoints: " << std::endl;
    for (std::vector<double> p : points) {
        std::cout << "\t\t" << p[0] << ", " << p[1] << ", " << p[2] << std::endl;
    }

    while(true) {

        std::cin >> message;
        std::string message_type = message["type"].template get<std::string>();

        if(message_type.compare("run-query") == 0) {
            std::cout << "\tradius: " << message["radius"].template get<int>() << std::endl;

            // Read the weights into a vector
            std::vector<double> weights = message["weights"].template get<std::vector<double>>();

            std::cout << "\tweights: " << std::endl;
            for (double w : weights) {
                std::cout << "\t\t" << w << std::endl;
            }
        }
    }
    
    // TODO: For any extra argument parsing use this
    // optind is for the extra arguments
    // which are not parsed
    // for(; optind < argc; optind++){     
    //     printf("extra arguments: %s\n", argv[optind]); 
    // }
      
    return 0;
}