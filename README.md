# Security-Model



Setup Instructions for All Team Members

Each feature should live inside its own folder under modules/. 
Example: modules/altercation_detector/

Implement your detection logic in a file called inference.py. Your main class should extend the shared base class: MonitoringModule (already in core/module_interface.py)

```
Your run() function must return:
{
    "status": "alert" or "normal",
    "confidence": float,
    "details": str,
    "module": "your_module_name"
}
```

Don’t handle video input or UI inside your module. Just process and return results.

You can test your module with the shared main.py script that runs all modules on webcam input.