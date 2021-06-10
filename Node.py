# from abc import ABC, abstractmethod

class Node:

    def __init__(self, initParams=None, time_series_length=None, idx=None):

        self.time_series_length = time_series_length
        self.params = {}
        self.idx = idx          # id is inbuilt. Dont use
        self.type = None

        if initParams is not None:
            self.setParamsFromDict(initParams)

    def setType(self, ty):
        self.type = ty

    def addParam(self, param):
        self.params[param] = [None for _ in range(self.time_series_length)]

    def addStaticParam(self, param, val):
        self.params[param] = val

    def setParam(self, paramName, t, val):
        self.params[paramName][t] = val


    def getParam(self, paramName, t):
        return self.params[paramName][t]


    def setParamsFromDict(self, dic):
        for par in dic:
            self.params[par] = dic[par]
        # self.params = dic
