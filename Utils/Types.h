//
// Created by auser on 2/4/25.
//

#ifndef COLLECTION_TYPES_H
#define COLLECTION_TYPES_H

template <typename T>
struct isPointer {
    static constexpr bool value = false;
};

template <typename T>
struct isPointer<T*> {
    static constexpr bool value = true;
};

template <typename T>
struct isReference {
    static constexpr bool value = false;
};

template <typename T>
struct isReference<T&> {
    static constexpr bool value = true;
};

template <typename T>
struct isReference<T&&> {
    static constexpr bool value = true;
};

template <typename T>
struct isLValueReference {
    static constexpr bool value = false;
};

template <typename T>
struct isLValueReference<T&> {
    static constexpr bool value = true;
};

template <typename T>
struct isRValueReference {
    static constexpr bool value = false;
};

template <typename T>
struct isRValueReference<T&&> {
    static constexpr bool value = true;
};

template <bool Condition, typename T = void>
struct enableIf {};

template <typename T>
struct enableIf<true, T> {
    using type = T;
};

template <typename T>
struct removeReference {
    using type = T;
};

template <typename T>
struct removeReference<T&> {
    using type = T;
};

template <typename T>
struct removeReference<T&&> {
    using type = T;
};

template <typename T>
struct removePointer {
    using type = T;
};

template <typename T>
struct removePointer<T*> {
    using type = T;
};

template <typename T>
struct removeConst {
    using type = T;
};

template <typename T>
struct removeConst<const T> {
    using type = T;
};

template <typename T>
removeReference<T>::type&& move( T&& arg ) {
    return static_cast<removeReference<T>::type&&>(arg);
}

template <typename T>
T&& forward(typename std::remove_reference<T>::type& arg) {
    return static_cast<T&&>(arg);
}

template <typename T>
T&& forward(typename std::remove_reference<T>::type&& arg) {
    return static_cast<T&&>(arg);
}

#endif //COLLECTION_TYPES_H
