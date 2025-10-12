//
// Created by auser on 12/30/24.
//

#ifndef COLLECTION_OUTPUTSTREAM_H
#define COLLECTION_OUTPUTSTREAM_H
#include <iostream>

template <typename StreamType>
class OutputStream {
private:
    StreamType* outputStream;

public:
    OutputStream(): outputStream( &std::cout ) {}
    OutputStream( StreamType* outputStream ): outputStream( outputStream ) {};
    ~OutputStream() = default;
    template <typename T>
    OutputStream& operator<<(const T& value) {
        *outputStream << value;
        return *this;
    }
};


#endif //COLLECTION_OUTPUTSTREAM_H
