#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <pthread.h>
#include <stdbool.h>

// Task function signature
typedef void (*task_func_t)(void* arg);

// Task structure
typedef struct task_t {
  task_func_t function;
  void* argument;
  struct task_t* next;
} task_t;

// Thread pool structure
typedef struct {
  pthread_t* threads;
  int thread_count;
  
  task_t* task_queue_head;
  task_t* task_queue_tail;
  
  pthread_mutex_t queue_mutex;
  pthread_cond_t queue_cond;
  pthread_cond_t done_cond;
  
  int active_tasks;
  bool shutdown;
} thread_pool_t;

/**
 * Create a thread pool with specified number of worker threads.
 * @param num_threads Number of worker threads to create
 * @return Pointer to thread pool, or NULL on error
 */
thread_pool_t* thread_pool_create(int num_threads);

/**
 * Add a task to the thread pool's queue.
 * @param pool Thread pool
 * @param function Function to execute
 * @param arg Argument to pass to function
 * @return true on success, false on error
 */
bool thread_pool_add_task(thread_pool_t* pool, task_func_t function, void* arg);

/**
 * Wait for all tasks to complete.
 * @param pool Thread pool
 */
void thread_pool_wait(thread_pool_t* pool);

/**
 * Destroy the thread pool and free all resources.
 * @param pool Thread pool to destroy
 */
void thread_pool_destroy(thread_pool_t* pool);

#endif // THREAD_POOL_H
