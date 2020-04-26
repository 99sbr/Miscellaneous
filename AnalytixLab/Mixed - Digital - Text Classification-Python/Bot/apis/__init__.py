from flask_restplus import Api
from flask import Blueprint

blueprint =Blueprint ('api',__name__)
from .namespace import api as ns1

api = Api(blueprint)
api.add_namespace(ns1)