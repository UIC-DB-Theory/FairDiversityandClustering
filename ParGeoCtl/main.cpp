#include <iostream>
#include <unistd.h>
#include "json.hpp"

using json = nlohmann::json;

const int READ_FROM_FILE = 100;
const int READ_JSON_STDIN = 200;

int main(int argc, char *argv[]) {
    
    int mode;
    
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
                json ex1 =  json::parse(optarg);
                std::cout << "\ttype: " << ex1["type"].template get<std::string>() << std::endl;
                break; 
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