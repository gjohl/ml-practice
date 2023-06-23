# Software Architecture Notes

Notes from udemy course https://www.udemy.com/course/the-complete-guide-to-becoming-a-software-architect/


## 1. What is a software architect
A developer knows what can be done, an architect knows what should be done. 
How do we use technology to meet business requirements. 

General system requirements:
- Fast
- Secure
- Reliable
- Easy to maintain

Architect's need to know how to code for:
- Architecture's trustworthiness
- Support developers
- Respect of developers


## 2. The architect's mindset
- Understand the business - Strengths, weaknesses, compettions, growth strategy.
- Define the system's goals - Goals are not requirements. Goals describe the effect on the organisation, requirements describe what the system should do.
- Work for your client's clients - Prioritise the end user.
- Talk to the right people with the right language - What is the thing that really matters to the person I'm talking to?
Project managers care about how things affect deadlines, developers care about the technologies used, CEOs care about business continuity and bottom line.


## 3. The architecture process
1. Understand the system requirements - What the system should do.
2. Understand the non-functional requirements - Technical and service level attributes, e.g. number of users, loads, volumes, performance.
3. Map the components - Understand the system functionality and communicate this to your client. 
Completely non-technical at this point, no mention of specific technologies. A diagram of high level components helps.
4. Select the technology stack - Backend, frontend, data store.
5. Design the architecture
6. Write the architecture document
7. Support the team

Include developers in non-functional requirements and architecture design for two reasons:
1. Learn about unknown scenarios early
2. Create ambassadors


## 4. System requirements
The 2 types of requirements: functional and non-functional.

### 4.1. Functional requirements
"What the system should do"

- Business flows
- Business services
- User interfaces

### 4.2. Non-functional requirements
"What the system should deal with"

- Performance - latency and throughput
  - Latency - How long does it take to perform a single task?
  - Throughput - How many tasks can be performed in a given time unit? 
- Load - Quantity of work without crashing. Determines the availability of the system. Always look at peak (worst case) numbers.
- Data volume - How much data will the system accumulate over time?
  - Data required on day one
  - Data growth (say, annually)
- Concurrent users - How many users will be using the system? This includes "dead times" which differentiates it from the load requirement.
Rule of thumb is concurrent_users = load * 10
- SLA - Service Level Agreement for uptime.

The non-functional requirements are generally the more important in determining the architecture.
The client will generally need guiding towards sensible values, otherwise they just want as much load as possible,
as much uptime as possible etc.


## 5. Application types
The application type should be established early based on the use case and expected user interaction.

- Web apps 
  - Serve html pages.
  - UI; user-initiated actions; large scale; short, focused actions.
  - Request-response model.
- Web API 
  - Serve data (often JSON).
  - Data retrieval and storage; client-initiated actions; large scale; short, focused actions.
  - Request-response model.
- Mobile
  - Require user interaction; frontend for web API; location-based.
- Console
  - No UI; limited interation; long-running processes; short actions for power users.
  - Require technical knowledge.
- Service
  - No UI; managed by service manager; long-running processes.
- Desktop
  - All resources on the PC; UI; user-centric actions.
- Function-as-a-service
  - AWS lambda


## 6. Select technology stack
Considerations:
- Appropriate for the task
- Community - e.g. stack overflow activity
- Popularity - google trends over 2 years

### 6.1. Backend technology
Covers web app, web API, console, service.

Options: .NET, JAva, node.js, PHP, Python
![img.png](images/software_architect/backend_tech.png)

### 6.2. Frontend technology
Covers:
- Web app - Angular, React
- Mobile - Native (Swift, Java/Kotlin), Xamarin, React Native
- Desktop - depends on target OS

### 6.3. Data store technology
SQL - small, structured data
- Relational tables
- Transactions
  - Atomicity
  - Consistency
  - Isolation
  - Durability
- Querying language is universal

NoSQL - huge, unstructured or semi-structured data
- Emphasis on scale and performance.
- Schema-less, with entities stored as JSON.
- Eventual consistency - data can be temporarily inconsistent
- No universal querying language


# 7. The *-ilities
Quality attributes that describe technical capabilities to fulfill the non-functional requirements.