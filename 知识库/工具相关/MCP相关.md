# mcp相关的内容
## 为什么选择MCP

那么，为什么要选择 MCP，或者说为什么要选择模型上下文协议 (Model Context Protocol) 呢？
我们常说，模型的表现好坏取决于提供给它们的上下文。
你可能拥有一个处于技术前沿、非常智能的模型，
但如果它无法连接到外部世界并获取必要的数据和上下文，它就无法发挥其应有的全部潜力。

## MCP与Function Calling总结

### 核心区别

- Function calling 确实做的是把"USB-B变成了Type-C"的工作 - 它解决了"连接问题"，让AI模型能够调用外部功能，建立了模型与工具之间的基本连接能力。

- MCP(Model Context Protocol) 则进一步做的是"把充电功率统一了"的工作 - 它在已有连接能力的基础上，通过标准化协议提高了连接的效率、一致性和可扩展性。

### 比喻解释

- Function calling解决了"能不能用"的问题(就像Type-C解决了能否物理连接)

- MCP解决了"用得好不好"的问题(就像统一充电协议解决了充电效率和兼容性)

## MCP 与 rest

> REST协议标准化了Web应用程序与后端通信的方式，MCP同样标准化了AI应用程序与数据源连接的方式。

### 具体例子：

REST标准化前，每个Web服务可能有完全不同的调用方式：
- 服务A: `executeQuery("SELECT * FROM users")`
- 服务B: `findAllPeople()`
- 服务C: `get_all_users_list()`

REST标准化后，所有服务采用统一方式：
- `GET /users`

同样，MCP标准化前，AI与不同工具的交互可能各不相同：
- 工具A: 自定义JSON格式
- 工具B: 自定义API调用
- 工具C: 特定于供应商的function calling格式

MCP标准化后，所有交互采用统一的方式，如：
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_data",
    "arguments": {...}
  }
}
```

REST通过定义统一的资源表示、操作方法和通信协议，大大简化了Web应用与后端系统的交互。类似地，MCP通过定义统一的工具调用、资源访问和通信格式，简化了AI模型与各种工具和数据源的交互。

这两种标准化方式都将复杂的集成问题转变为遵循共同协议的简单问题，极大地提高了开发效率和系统互操作性。