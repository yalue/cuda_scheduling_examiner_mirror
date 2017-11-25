
#ifndef NVXIO_EXPORT_H
#define NVXIO_EXPORT_H

#ifdef NVXIO_STATIC_DEFINE
#  define NVXIO_EXPORT
#  define NVXIO_NO_EXPORT
#else
#  ifndef NVXIO_EXPORT
#    ifdef nvxio_EXPORTS
        /* We are building this library */
#      define NVXIO_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define NVXIO_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef NVXIO_NO_EXPORT
#    define NVXIO_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef NVXIO_DEPRECATED
#  define NVXIO_DEPRECATED __attribute__ ((__deprecated__))
#  define NVXIO_DEPRECATED_EXPORT NVXIO_EXPORT __attribute__ ((__deprecated__))
#  define NVXIO_DEPRECATED_NO_EXPORT NVXIO_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define NVXIO_NO_DEPRECATED
#endif

#endif
