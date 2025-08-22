
CC		:= gcc
EXE		:= correction_test.exe

SRCS	= $(wildcard *.c)
OBJS	= $(SRCS:.c=.o)
DEPEND	= $(OBJS:%.o=%.d)

$(EXE): $(OBJS)
	$(CC) -o $@ $(OBJS) 2>&1
	@test -x $@

-include $(DEPEND)

.c.o:
	$(CC) $(CFLAGS) -MMD -c $< -o $@

clean:
	rm -f $(OBJS) $(EXE) $(DEPEND) *.out
