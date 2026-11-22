PY=/root/jpp_lens/.venv/bin/python
$PY /workspace/o-lens-improvement/scripts/dp1/test_repeat.py fla > /root/dp1/test_repeat.txt 2>/root/dp1/test_repeat.err
$PY /workspace/o-lens-improvement/scripts/dp1/test_repeat.py nofla >> /root/dp1/test_repeat.txt 2>>/root/dp1/test_repeat.err
touch /root/dp1/test_repeat.done
