# Final-Project
##When running 'streamlit app.py (from src/gui directory), if
Traceback (most recent call last):
  File "/Users/gteitel/Desktop/Final_project/venv/bin/streamlit", line 5, in <module>
    from streamlit.cli import main
  File "/Users/gteitel/Desktop/Final_project/venv/lib/python3.9/site-packages/streamlit/__init__.py", line 48, in <module>
    from streamlit.proto.RootContainer_pb2 import RootContainer
  File "/Users/gteitel/Desktop/Final_project/venv/lib/python3.9/site-packages/streamlit/proto/RootContainer_pb2.py", line 33, in <module>
    _descriptor.EnumValueDescriptor(
  File "/Users/gteitel/Desktop/Final_project/venv/lib/python3.9/site-packages/google/protobuf/descriptor.py", line 920, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
 
'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python'
