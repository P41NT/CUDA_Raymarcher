NVCC        := nvcc
CFLAGS      := -O2 -arch=sm_75 -Iinclude

LDFLAGS     := -lraylib

BUILDDIR    := builds
SRCDIR      := src

SOURCES     := $(wildcard $(SRCDIR)/*.cu)
OBJECTS     := $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.o, $(SOURCES))
TARGET      := $(BUILDDIR)/main

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(CFLAGS) -dc $< -o $@

clean:
	rm -f $(BUILDDIR)/*.o $(TARGET)

run: all
	./$(TARGET)

