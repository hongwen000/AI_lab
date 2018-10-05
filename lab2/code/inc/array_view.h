//
// Created by 李新锐 on 04/10/2018.
//

#ifndef DECISIONTREE_ARRAY_VIEW_H
#define DECISIONTREE_ARRAY_VIEW_H

#include "vector"
#include "string"
#include "functional"

template <typename T>
class array_view {
    std::vector<std::reference_wrapper<const std::vector<T>>> view;
public:
    explicit array_view(const std::vector<std::vector<T>>& vec)
    {
        for(const auto& v : vec)
        {
            view.push_back(std::cref(v));
        }
    }
    array_view(const std::vector<std::vector<T>>& vec, const std::set<size_t>& ilist)
    {
        for(const auto& i: ilist)
        {
            if(i < vec.size())
                view.push_back(std::cref(vec[i]));
            else
                throw(std::out_of_range("array_view ~ ctor ~ vec.size = " + std::to_string(vec.size()) + " ~ i = " + std::to_string(i)));
        }
    }
    array_view(const array_view<T>& father, const std::set<size_t>& ilist)
    {
        for(const auto& i: ilist)
        {
            if(i < father.size())
                view.push_back(std::cref(father[i]));
            else
                throw(std::out_of_range("array_view ~ ctor ~ father.size = " + std::to_string(father.size()) + " ~ i = " + std::to_string(i)));
        }

    }

    decltype(view.size()) size() const
    {
        return view.size();
    }
struct iterator {
        explicit iterator(typename std::vector<std::reference_wrapper<const std::vector<T>>>::iterator _it): it(_it){};
        typename std::vector<std::reference_wrapper<const std::vector<T>>>::iterator it;
        const std::vector<T>&operator*() const
        {
            return (*it).get();
        }
        iterator& operator++() {
            ++it;
            return *this;
        }
        bool operator==(const iterator& rhs) const
        {
            return it == rhs.it;
        }
        bool operator!=(const iterator& rhs) const
        {
            return it != rhs.it;
        }
    };

    iterator begin()
    {
        return iterator{view.begin()};
    }

    iterator end()
    {
        return iterator{view.end()};
    }

    iterator begin() const
    {
        return iterator{view.begin()};
    }

    iterator end() const
    {
        return iterator{view.end()};
    }
    const std::vector<T> operator[](unsigned long i) const
    {
        return view[i].get();
    }

    std::vector<T> operator[](unsigned long i)
    {
        return view[i].get();
    }

};

#endif //DECISIONTREE_ARRAY_VIEW_H
