#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mexico_dollar.py - What is the dollar of Mexico?
author: Bill Thompson
license: GPL 3
copyright: 2023-04-27
"""
from hdc import hdv, bundle, bind, cos

def main():
    U = hdv()  # USA
    M = hdv()  # Mexico
    D = hdv()  # dollar
    P = hdv()  # peso

    X = hdv()  # country
    Y = hdv()  # currency

    A = bundle(bind(X, U), bind(Y, D))
    B = bundle(bind(X, M), bind(Y, P))

    # bind(D, A) = bundle(bind(D, bind(X, U)), bind(D, bind(Y, D))) = bundle(bind(D, bind(X, U)), Y) ~ Y
    dollar_role_us = bind(D, A)  
    print(cos(dollar_role_us, Y)) # dollar is currency of US
    print(cos(dollar_role_us, B)) # dollar is not currency of Mexico

    # bind(dollar_role_us, B) ~ P
    dollar_of_mexico = bind(dollar_role_us, B)
    print(cos(dollar_of_mexico, P))  # peso is dollar of mexico
    print(cos(dollar_of_mexico, D))  # dollar is not currency of Mexico
    


if __name__ == "__main__":
    main()
