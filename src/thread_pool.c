#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>

#include "../include/thread_pool.h"

// Worker thread function
static void* worker_thread(void* arg) {
  thread_pool_t* pool = (thread_pool_t*)arg;
  
  while (true) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    // Wait for a task or shutdown signal
    while (pool->task_queue_head == NULL && !pool->shutdown) {
      pthread_cond_wait(&pool->queue_cond, &pool->queue_mutex);
    }
    
    // Check for shutdown
    if (pool->shutdown && pool->task_queue_head == NULL) {
      pthread_mutex_unlock(&pool->queue_mutex);
      break;
    }
    
    // Dequeue task
    task_t* task = pool->task_queue_head;
    if (task != NULL) {
      pool->task_queue_head = task->next;
      if (pool->task_queue_head == NULL) {
        pool->task_queue_tail = NULL;
      }
      pool->active_tasks++;
    }
    
    pthread_mutex_unlock(&pool->queue_mutex);
    
    // Execute task
    if (task != NULL) {
      task->function(task->argument);
      free(task);
      
      pthread_mutex_lock(&pool->queue_mutex);
      pool->active_tasks--;
      if (pool->active_tasks == 0 && pool->task_queue_head == NULL) {
        pthread_cond_broadcast(&pool->done_cond);
      }
      pthread_mutex_unlock(&pool->queue_mutex);
    }
  }
  
  return NULL;
}

thread_pool_t* thread_pool_create(int num_threads) {
  if (num_threads <= 0) {
    return NULL;
  }
  
  thread_pool_t* pool = (thread_pool_t*)malloc(sizeof(thread_pool_t));
  if (pool == NULL) {
    return NULL;
  }
  
  pool->thread_count = num_threads;
  pool->task_queue_head = NULL;
  pool->task_queue_tail = NULL;
  pool->active_tasks = 0;
  pool->shutdown = false;
  
  // Initialize mutex and condition variables
  if (pthread_mutex_init(&pool->queue_mutex, NULL) != 0) {
    free(pool);
    return NULL;
  }
  
  if (pthread_cond_init(&pool->queue_cond, NULL) != 0) {
    pthread_mutex_destroy(&pool->queue_mutex);
    free(pool);
    return NULL;
  }
  
  if (pthread_cond_init(&pool->done_cond, NULL) != 0) {
    pthread_cond_destroy(&pool->queue_cond);
    pthread_mutex_destroy(&pool->queue_mutex);
    free(pool);
    return NULL;
  }
  
  // Allocate thread array
  pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
  if (pool->threads == NULL) {
    pthread_cond_destroy(&pool->done_cond);
    pthread_cond_destroy(&pool->queue_cond);
    pthread_mutex_destroy(&pool->queue_mutex);
    free(pool);
    return NULL;
  }
  
  // Create worker threads
  for (int i = 0; i < num_threads; i++) {
    if (pthread_create(&pool->threads[i], NULL, worker_thread, pool) != 0) {
      // Failed to create thread, cleanup
      pool->shutdown = true;
      pthread_cond_broadcast(&pool->queue_cond);
      for (int j = 0; j < i; j++) {
        pthread_join(pool->threads[j], NULL);
      }
      free(pool->threads);
      pthread_cond_destroy(&pool->done_cond);
      pthread_cond_destroy(&pool->queue_cond);
      pthread_mutex_destroy(&pool->queue_mutex);
      free(pool);
      return NULL;
    }
  }
  
  return pool;
}

bool thread_pool_add_task(thread_pool_t* pool, task_func_t function, void* arg) {
  if (pool == NULL || function == NULL) {
    return false;
  }
  
  task_t* task = (task_t*)malloc(sizeof(task_t));
  if (task == NULL) {
    return false;
  }
  
  task->function = function;
  task->argument = arg;
  task->next = NULL;
  
  pthread_mutex_lock(&pool->queue_mutex);
  
  if (pool->shutdown) {
    pthread_mutex_unlock(&pool->queue_mutex);
    free(task);
    return false;
  }
  
  // Enqueue task
  if (pool->task_queue_tail == NULL) {
    pool->task_queue_head = task;
    pool->task_queue_tail = task;
  } else {
    pool->task_queue_tail->next = task;
    pool->task_queue_tail = task;
  }
  
  pthread_cond_signal(&pool->queue_cond);
  pthread_mutex_unlock(&pool->queue_mutex);
  
  return true;
}

void thread_pool_wait(thread_pool_t* pool) {
  if (pool == NULL) {
    return;
  }
  
  pthread_mutex_lock(&pool->queue_mutex);
  
  while (pool->active_tasks > 0 || pool->task_queue_head != NULL) {
    pthread_cond_wait(&pool->done_cond, &pool->queue_mutex);
  }
  
  pthread_mutex_unlock(&pool->queue_mutex);
}

void thread_pool_destroy(thread_pool_t* pool) {
  if (pool == NULL) {
    return;
  }
  
  pthread_mutex_lock(&pool->queue_mutex);
  pool->shutdown = true;
  pthread_cond_broadcast(&pool->queue_cond);
  pthread_mutex_unlock(&pool->queue_mutex);
  
  // Join all threads
  for (int i = 0; i < pool->thread_count; i++) {
    pthread_join(pool->threads[i], NULL);
  }
  
  // Free remaining tasks in queue
  task_t* task = pool->task_queue_head;
  while (task != NULL) {
    task_t* next = task->next;
    free(task);
    task = next;
  }
  
  free(pool->threads);
  pthread_cond_destroy(&pool->done_cond);
  pthread_cond_destroy(&pool->queue_cond);
  pthread_mutex_destroy(&pool->queue_mutex);
  free(pool);
}
