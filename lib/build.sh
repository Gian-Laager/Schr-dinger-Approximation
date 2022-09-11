#! /bin/bash

go get main
go build -o libairy.a -buildmode=c-archive main.go
